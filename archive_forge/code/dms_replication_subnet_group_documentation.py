from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule

    Compares the dict obtained from the describe function and
    what we are reading from the values in the template We can
    never compare passwords as boto3's method for describing
    a DMS endpoint does not return the value for
    the password for security reasons ( I assume )
    