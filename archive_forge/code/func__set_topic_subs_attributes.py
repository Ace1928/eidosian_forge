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