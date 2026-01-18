from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class AmazonBucket:

    def __init__(self, module, client):
        self.module = module
        self.client = client
        self.bucket_name = module.params['bucket_name']
        self.check_mode = module.check_mode
        self._full_config_cache = None

    def full_config(self):
        if self._full_config_cache is None:
            self._full_config_cache = dict(QueueConfigurations=[], TopicConfigurations=[], LambdaFunctionConfigurations=[])
            try:
                config_lookup = self.client.get_bucket_notification_configuration(Bucket=self.bucket_name)
            except (ClientError, BotoCoreError) as e:
                self.module.fail_json(msg=f'{e}')
            if config_lookup.get('QueueConfigurations'):
                for queue_config in config_lookup.get('QueueConfigurations'):
                    self._full_config_cache['QueueConfigurations'].append(Config.from_api(queue_config))
            if config_lookup.get('TopicConfigurations'):
                for topic_config in config_lookup.get('TopicConfigurations'):
                    self._full_config_cache['TopicConfigurations'].append(Config.from_api(topic_config))
            if config_lookup.get('LambdaFunctionConfigurations'):
                for function_config in config_lookup.get('LambdaFunctionConfigurations'):
                    self._full_config_cache['LambdaFunctionConfigurations'].append(Config.from_api(function_config))
        return self._full_config_cache

    def current_config(self, config_name):
        for target_configs in self.full_config():
            for config in self.full_config()[target_configs]:
                if config.raw['Id'] == config_name:
                    return config

    def apply_config(self, desired):
        configs = dict(QueueConfigurations=[], TopicConfigurations=[], LambdaFunctionConfigurations=[])
        for target_configs in self.full_config():
            for config in self.full_config()[target_configs]:
                if config.name != desired.raw['Id']:
                    configs[target_configs].append(config.raw)
        if self.module.params.get('queue_arn'):
            configs['QueueConfigurations'].append(desired.raw)
        if self.module.params.get('topic_arn'):
            configs['TopicConfigurations'].append(desired.raw)
        if self.module.params.get('lambda_function_arn'):
            configs['LambdaFunctionConfigurations'].append(desired.raw)
        self._upload_bucket_config(configs)
        return configs

    def delete_config(self, desired):
        configs = dict(QueueConfigurations=[], TopicConfigurations=[], LambdaFunctionConfigurations=[])
        for target_configs in self.full_config():
            for config in self.full_config()[target_configs]:
                if config.name != desired.raw['Id']:
                    configs[target_configs].append(config.raw)
        self._upload_bucket_config(configs)
        return configs

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