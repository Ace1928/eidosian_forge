from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class CodeCommit(object):

    def __init__(self, module=None):
        self._module = module
        self._client = self._module.client('codecommit')
        self._check_mode = self._module.check_mode

    def process(self):
        result = dict(changed=False)
        if self._module.params['state'] == 'present':
            if not self._repository_exists():
                if not self._check_mode:
                    result = self._create_repository()
                result['changed'] = True
            else:
                metadata = self._get_repository()['repositoryMetadata']
                if not metadata.get('repositoryDescription'):
                    metadata['repositoryDescription'] = ''
                if metadata['repositoryDescription'] != self._module.params['description']:
                    if not self._check_mode:
                        self._update_repository()
                    result['changed'] = True
                result.update(self._get_repository())
        if self._module.params['state'] == 'absent' and self._repository_exists():
            if not self._check_mode:
                result = self._delete_repository()
            result['changed'] = True
        return result

    def _repository_exists(self):
        try:
            paginator = self._client.get_paginator('list_repositories')
            for page in paginator.paginate():
                repositories = page['repositories']
                for item in repositories:
                    if self._module.params['name'] in item.values():
                        return True
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self._module.fail_json_aws(e, msg="couldn't get repository")
        return False

    def _get_repository(self):
        try:
            result = self._client.get_repository(repositoryName=self._module.params['name'])
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self._module.fail_json_aws(e, msg="couldn't get repository")
        return result

    def _update_repository(self):
        try:
            result = self._client.update_repository_description(repositoryName=self._module.params['name'], repositoryDescription=self._module.params['description'])
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self._module.fail_json_aws(e, msg="couldn't create repository")
        return result

    def _create_repository(self):
        try:
            result = self._client.create_repository(repositoryName=self._module.params['name'], repositoryDescription=self._module.params['description'])
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self._module.fail_json_aws(e, msg="couldn't create repository")
        return result

    def _delete_repository(self):
        try:
            result = self._client.delete_repository(repositoryName=self._module.params['name'])
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self._module.fail_json_aws(e, msg="couldn't delete repository")
        return result