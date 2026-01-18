from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.jittered_backoff(retries=3, delay=5)
def _delete_saml_provider(self, arn):
    return self.conn.delete_saml_provider(SAMLProviderArn=arn)