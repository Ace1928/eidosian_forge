from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _upload_bucket_config(self, configs):
    api_params = dict(Bucket=self.bucket_name, NotificationConfiguration=dict())
    for target_configs in configs:
        if len(configs[target_configs]) > 0:
            api_params['NotificationConfiguration'][target_configs] = configs[target_configs]
    if not self.check_mode:
        try:
            self.client.put_bucket_notification_configuration(**api_params)
        except (ClientError, BotoCoreError) as e:
            self.module.fail_json(msg=f'{e}')