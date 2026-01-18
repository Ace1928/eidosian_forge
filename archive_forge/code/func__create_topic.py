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