from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
@AWSRetry.jittered_backoff(retries=3, delay=5)
def _create_saml_provider(self, metadata, name):
    return self.conn.create_saml_provider(SAMLMetadataDocument=metadata, Name=name)