from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class SAMLProviderManager:
    """Handles SAML Identity Provider configuration"""

    def __init__(self, module):
        self.module = module
        try:
            self.conn = module.client('iam')
        except botocore.exceptions.ClientError as e:
            self.module.fail_json_aws(e, msg='Unknown AWS SDK error')

    @AWSRetry.jittered_backoff(retries=3, delay=5)
    def _list_saml_providers(self):
        return self.conn.list_saml_providers()

    @AWSRetry.jittered_backoff(retries=3, delay=5)
    def _get_saml_provider(self, arn):
        return self.conn.get_saml_provider(SAMLProviderArn=arn)

    @AWSRetry.jittered_backoff(retries=3, delay=5)
    def _update_saml_provider(self, arn, metadata):
        return self.conn.update_saml_provider(SAMLProviderArn=arn, SAMLMetadataDocument=metadata)

    @AWSRetry.jittered_backoff(retries=3, delay=5)
    def _create_saml_provider(self, metadata, name):
        return self.conn.create_saml_provider(SAMLMetadataDocument=metadata, Name=name)

    @AWSRetry.jittered_backoff(retries=3, delay=5)
    def _delete_saml_provider(self, arn):
        return self.conn.delete_saml_provider(SAMLProviderArn=arn)

    def _get_provider_arn(self, name):
        providers = self._list_saml_providers()
        for p in providers['SAMLProviderList']:
            provider_name = p['Arn'].split('/', 1)[1]
            if name == provider_name:
                return p['Arn']
        return None

    def create_or_update_saml_provider(self, name, metadata):
        if not metadata:
            self.module.fail_json(msg='saml_metadata_document must be defined for present state')
        res = {'changed': False}
        try:
            arn = self._get_provider_arn(name)
        except (botocore.exceptions.ValidationError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg=f"Could not get the ARN of the identity provider '{name}'")
        if arn:
            try:
                resp = self._get_saml_provider(arn)
            except (botocore.exceptions.ValidationError, botocore.exceptions.ClientError) as e:
                self.module.fail_json_aws(e, msg=f"Could not retrieve the identity provider '{name}'")
            if metadata.strip() != resp['SAMLMetadataDocument'].strip():
                res['changed'] = True
                if not self.module.check_mode:
                    try:
                        resp = self._update_saml_provider(arn, metadata)
                        res['saml_provider'] = self._build_res(resp['SAMLProviderArn'])
                    except botocore.exceptions.ClientError as e:
                        self.module.fail_json_aws(e, msg=f"Could not update the identity provider '{name}'")
            else:
                res['saml_provider'] = self._build_res(arn)
        else:
            res['changed'] = True
            if not self.module.check_mode:
                try:
                    resp = self._create_saml_provider(metadata, name)
                    res['saml_provider'] = self._build_res(resp['SAMLProviderArn'])
                except botocore.exceptions.ClientError as e:
                    self.module.fail_json_aws(e, msg=f"Could not create the identity provider '{name}'")
        self.module.exit_json(**res)

    def delete_saml_provider(self, name):
        res = {'changed': False}
        try:
            arn = self._get_provider_arn(name)
        except (botocore.exceptions.ValidationError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg=f"Could not get the ARN of the identity provider '{name}'")
        if arn:
            res['changed'] = True
            if not self.module.check_mode:
                try:
                    self._delete_saml_provider(arn)
                except botocore.exceptions.ClientError as e:
                    self.module.fail_json_aws(e, msg=f"Could not delete the identity provider '{name}'")
        self.module.exit_json(**res)

    def _build_res(self, arn):
        saml_provider = self._get_saml_provider(arn)
        return {'arn': arn, 'metadata_document': saml_provider['SAMLMetadataDocument'], 'create_date': saml_provider['CreateDate'].isoformat(), 'expire_date': saml_provider['ValidUntil'].isoformat()}