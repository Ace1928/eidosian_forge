from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule

    Copies an EC2 Snapshot to another region

    module : AnsibleAWSModule object
    ec2: ec2 connection object
    