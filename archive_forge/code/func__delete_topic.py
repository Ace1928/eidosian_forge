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
def _delete_topic(self):
    self.topic_deleted = True
    if not self.check_mode:
        try:
            self.connection.delete_topic(TopicArn=self.topic_arn)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.module.fail_json_aws(e, msg=f"Couldn't delete topic {self.topic_arn}")
    return True