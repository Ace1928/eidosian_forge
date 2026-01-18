import json
from ansible_collections.amazon.aws.plugins.module_utils.arn import parse_aws_arn
from ansible_collections.amazon.aws.plugins.module_utils.policy import compare_policies
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
from ansible_collections.community.aws.plugins.module_utils.sns import canonicalize_endpoint
from ansible_collections.community.aws.plugins.module_utils.sns import compare_delivery_policies
from ansible_collections.community.aws.plugins.module_utils.sns import get_info
from ansible_collections.community.aws.plugins.module_utils.sns import list_topic_subscriptions
from ansible_collections.community.aws.plugins.module_utils.sns import list_topics
from ansible_collections.community.aws.plugins.module_utils.sns import topic_arn_lookup
from ansible_collections.community.aws.plugins.module_utils.sns import update_tags
class SnsTopicManager(object):
    """Handles SNS Topic creation and destruction"""

    def __init__(self, module, name, topic_type, state, display_name, policy, delivery_policy, subscriptions, purge_subscriptions, tags, purge_tags, content_based_deduplication, check_mode):
        self.connection = module.client('sns')
        self.module = module
        self.name = name
        self.topic_type = topic_type
        self.state = state
        self.display_name = display_name
        self.policy = policy
        self.delivery_policy = scrub_none_parameters(delivery_policy) if delivery_policy else None
        self.subscriptions = subscriptions
        self.subscriptions_existing = []
        self.subscriptions_deleted = []
        self.subscriptions_added = []
        self.subscriptions_attributes_set = []
        self.desired_subscription_attributes = dict()
        self.purge_subscriptions = purge_subscriptions
        self.content_based_deduplication = content_based_deduplication
        self.check_mode = check_mode
        self.topic_created = False
        self.topic_deleted = False
        self.topic_arn = None
        self.attributes_set = []
        self.tags = tags
        self.purge_tags = purge_tags

    def _create_topic(self):
        attributes = {}
        tags = []
        if self.topic_type == 'fifo':
            attributes['FifoTopic'] = 'true'
            if not self.name.endswith('.fifo'):
                self.name = self.name + '.fifo'
        if self.tags:
            tags = ansible_dict_to_boto3_tag_list(self.tags)
        if not self.check_mode:
            try:
                response = self.connection.create_topic(Name=self.name, Attributes=attributes, Tags=tags)
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                self.module.fail_json_aws(e, msg=f"Couldn't create topic {self.name}")
            self.topic_arn = response['TopicArn']
        return True

    def _set_topic_attrs(self):
        changed = False
        try:
            topic_attributes = self.connection.get_topic_attributes(TopicArn=self.topic_arn)['Attributes']
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.module.fail_json_aws(e, msg=f"Couldn't get topic attributes for topic {self.topic_arn}")
        if self.display_name and self.display_name != topic_attributes['DisplayName']:
            changed = True
            self.attributes_set.append('display_name')
            if not self.check_mode:
                try:
                    self.connection.set_topic_attributes(TopicArn=self.topic_arn, AttributeName='DisplayName', AttributeValue=self.display_name)
                except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                    self.module.fail_json_aws(e, msg="Couldn't set display name")
        if self.policy and compare_policies(self.policy, json.loads(topic_attributes['Policy'])):
            changed = True
            self.attributes_set.append('policy')
            if not self.check_mode:
                try:
                    self.connection.set_topic_attributes(TopicArn=self.topic_arn, AttributeName='Policy', AttributeValue=json.dumps(self.policy))
                except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                    self.module.fail_json_aws(e, msg="Couldn't set topic policy")
        if ('FifoTopic' in topic_attributes and topic_attributes['FifoTopic'] == 'true') and self.content_based_deduplication:
            enabled = 'true' if self.content_based_deduplication in 'enabled' else 'false'
            if enabled != topic_attributes['ContentBasedDeduplication']:
                changed = True
                self.attributes_set.append('content_based_deduplication')
                if not self.check_mode:
                    try:
                        self.connection.set_topic_attributes(TopicArn=self.topic_arn, AttributeName='ContentBasedDeduplication', AttributeValue=enabled)
                    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                        self.module.fail_json_aws(e, msg="Couldn't set content-based deduplication")
        if self.delivery_policy and ('DeliveryPolicy' not in topic_attributes or compare_delivery_policies(self.delivery_policy, json.loads(topic_attributes['DeliveryPolicy']))):
            changed = True
            self.attributes_set.append('delivery_policy')
            if not self.check_mode:
                try:
                    self.connection.set_topic_attributes(TopicArn=self.topic_arn, AttributeName='DeliveryPolicy', AttributeValue=json.dumps(self.delivery_policy))
                except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                    self.module.fail_json_aws(e, msg="Couldn't set topic delivery policy")
        return changed

    def _set_topic_subs(self):
        changed = False
        subscriptions_existing_list = set()
        desired_subscriptions = [(sub['protocol'], canonicalize_endpoint(sub['protocol'], sub['endpoint'])) for sub in self.subscriptions]
        for sub in list_topic_subscriptions(self.connection, self.module, self.topic_arn):
            sub_key = (sub['Protocol'], sub['Endpoint'])
            subscriptions_existing_list.add(sub_key)
            if self.purge_subscriptions and sub_key not in desired_subscriptions and (sub['SubscriptionArn'] not in ('PendingConfirmation', 'Deleted')):
                changed = True
                self.subscriptions_deleted.append(sub_key)
                if not self.check_mode:
                    try:
                        self.connection.unsubscribe(SubscriptionArn=sub['SubscriptionArn'])
                    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                        self.module.fail_json_aws(e, msg="Couldn't unsubscribe from topic")
        for protocol, endpoint in set(desired_subscriptions).difference(subscriptions_existing_list):
            changed = True
            self.subscriptions_added.append((protocol, endpoint))
            if not self.check_mode:
                try:
                    self.connection.subscribe(TopicArn=self.topic_arn, Protocol=protocol, Endpoint=endpoint)
                except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                    self.module.fail_json_aws(e, msg=f"Couldn't subscribe to topic {self.topic_arn}")
        return changed

    def _init_desired_subscription_attributes(self):
        for sub in self.subscriptions:
            sub_key = (sub['protocol'], canonicalize_endpoint(sub['protocol'], sub['endpoint']))
            tmp_dict = sub.get('attributes', {})
            for k, v in tmp_dict.items():
                tmp_dict[k] = str(v)
            self.desired_subscription_attributes[sub_key] = tmp_dict

    def _set_topic_subs_attributes(self):
        changed = False
        for sub in list_topic_subscriptions(self.connection, self.module, self.topic_arn):
            sub_key = (sub['Protocol'], sub['Endpoint'])
            sub_arn = sub['SubscriptionArn']
            if not self.desired_subscription_attributes.get(sub_key):
                continue
            try:
                sub_current_attributes = self.connection.get_subscription_attributes(SubscriptionArn=sub_arn)['Attributes']
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                self.module.fail_json_aws(e, f"Couldn't get subscription attributes for subscription {sub_arn}")
            raw_message = self.desired_subscription_attributes[sub_key].get('RawMessageDelivery')
            if raw_message is not None and 'RawMessageDelivery' in sub_current_attributes:
                if sub_current_attributes['RawMessageDelivery'].lower() != raw_message.lower():
                    changed = True
                    if not self.check_mode:
                        try:
                            self.connection.set_subscription_attributes(SubscriptionArn=sub_arn, AttributeName='RawMessageDelivery', AttributeValue=raw_message)
                        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                            self.module.fail_json_aws(e, "Couldn't set RawMessageDelivery subscription attribute")
        return changed

    def _delete_subscriptions(self):
        subscriptions = list_topic_subscriptions(self.connection, self.module, self.topic_arn)
        if not subscriptions:
            return False
        for sub in subscriptions:
            if sub['SubscriptionArn'] not in ('PendingConfirmation', 'Deleted'):
                self.subscriptions_deleted.append(sub['SubscriptionArn'])
                if not self.check_mode:
                    try:
                        self.connection.unsubscribe(SubscriptionArn=sub['SubscriptionArn'])
                    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                        self.module.fail_json_aws(e, msg="Couldn't unsubscribe from topic")
        return True

    def _delete_topic(self):
        self.topic_deleted = True
        if not self.check_mode:
            try:
                self.connection.delete_topic(TopicArn=self.topic_arn)
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                self.module.fail_json_aws(e, msg=f"Couldn't delete topic {self.topic_arn}")
        return True

    def _name_is_arn(self):
        return bool(parse_aws_arn(self.name))

    def ensure_ok(self):
        changed = False
        self.populate_topic_arn()
        if not self.topic_arn:
            changed = self._create_topic()
        if self.topic_arn in list_topics(self.connection, self.module):
            changed |= self._set_topic_attrs()
        elif self.display_name or self.policy or self.delivery_policy:
            self.module.fail_json(msg='Cannot set display name, policy or delivery policy for SNS topics not owned by this account')
        changed |= self._set_topic_subs()
        self._init_desired_subscription_attributes()
        if self.topic_arn in list_topics(self.connection, self.module):
            changed |= self._set_topic_subs_attributes()
        elif any(self.desired_subscription_attributes.values()):
            self.module.fail_json(msg='Cannot set subscription attributes for SNS topics not owned by this account')
        changed |= update_tags(self.connection, self.module, self.topic_arn)
        return changed

    def ensure_gone(self):
        changed = False
        self.populate_topic_arn()
        if self.topic_arn:
            if self.topic_arn not in list_topics(self.connection, self.module):
                self.module.fail_json(msg='Cannot use state=absent with third party ARN. Use subscribers=[] to unsubscribe')
            changed = self._delete_subscriptions()
            changed |= self._delete_topic()
        return changed

    def populate_topic_arn(self):
        if self._name_is_arn():
            self.topic_arn = self.name
            return
        name = self.name
        if self.topic_type == 'fifo' and (not name.endswith('.fifo')):
            name += '.fifo'
        self.topic_arn = topic_arn_lookup(self.connection, self.module, name)