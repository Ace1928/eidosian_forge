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