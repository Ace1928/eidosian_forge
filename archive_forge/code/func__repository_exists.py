from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
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