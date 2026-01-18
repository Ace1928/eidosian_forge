from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _get_provider_arn(self, name):
    providers = self._list_saml_providers()
    for p in providers['SAMLProviderList']:
        provider_name = p['Arn'].split('/', 1)[1]
        if name == provider_name:
            return p['Arn']
    return None