import json
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
class CloudWatchEventRule:

    def __init__(self, module, name, client, schedule_expression=None, event_pattern=None, description=None, role_arn=None):
        self.name = name
        self.client = client
        self.changed = False
        self.schedule_expression = schedule_expression
        self.event_pattern = event_pattern
        self.description = description
        self.role_arn = role_arn
        self.module = module

    def describe(self):
        """Returns the existing details of the rule in AWS"""
        try:
            rule_info = self.client.describe_rule(Name=self.name)
        except is_boto3_error_code('ResourceNotFoundException'):
            return {}
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg=f'Could not describe rule {self.name}')
        return camel_dict_to_snake_dict(rule_info)

    def put(self, enabled=True):
        """Creates or updates the rule in AWS"""
        request = {'Name': self.name, 'State': 'ENABLED' if enabled else 'DISABLED'}
        if self.schedule_expression:
            request['ScheduleExpression'] = self.schedule_expression
        if self.event_pattern:
            request['EventPattern'] = self.event_pattern
        if self.description:
            request['Description'] = self.description
        if self.role_arn:
            request['RoleArn'] = self.role_arn
        try:
            response = self.client.put_rule(**request)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg=f'Could not create/update rule {self.name}')
        self.changed = True
        return response

    def delete(self):
        """Deletes the rule in AWS"""
        self.remove_all_targets()
        try:
            response = self.client.delete_rule(Name=self.name)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg=f'Could not delete rule {self.name}')
        self.changed = True
        return response

    def enable(self):
        """Enables the rule in AWS"""
        try:
            response = self.client.enable_rule(Name=self.name)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg=f'Could not enable rule {self.name}')
        self.changed = True
        return response

    def disable(self):
        """Disables the rule in AWS"""
        try:
            response = self.client.disable_rule(Name=self.name)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg=f'Could not disable rule {self.name}')
        self.changed = True
        return response

    def list_targets(self):
        """Lists the existing targets for the rule in AWS"""
        try:
            targets = self.client.list_targets_by_rule(Rule=self.name)
        except is_boto3_error_code('ResourceNotFoundException'):
            return []
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg=f'Could not find target for rule {self.name}')
        return camel_dict_to_snake_dict(targets)['targets']

    def put_targets(self, targets):
        """Creates or updates the provided targets on the rule in AWS"""
        if not targets:
            return
        request = {'Rule': self.name, 'Targets': self._targets_request(targets)}
        try:
            response = self.client.put_targets(**request)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg=f'Could not create/update rule targets for rule {self.name}')
        self.changed = True
        return response

    def remove_targets(self, target_ids):
        """Removes the provided targets from the rule in AWS"""
        if not target_ids:
            return
        request = {'Rule': self.name, 'Ids': target_ids}
        try:
            response = self.client.remove_targets(**request)
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg=f'Could not remove rule targets from rule {self.name}')
        self.changed = True
        return response

    def remove_all_targets(self):
        """Removes all targets on rule"""
        targets = self.list_targets()
        return self.remove_targets([t['id'] for t in targets])

    def _targets_request(self, targets):
        """Formats each target for the request"""
        targets_request = []
        for target in targets:
            target_request = scrub_none_parameters(snake_dict_to_camel_dict(target, True))
            if target_request.get('Input', None):
                target_request['Input'] = _format_json(target_request['Input'])
            if target_request.get('InputTransformer', None):
                if target_request.get('InputTransformer').get('InputTemplate', None):
                    target_request['InputTransformer']['InputTemplate'] = _format_json(target_request['InputTransformer']['InputTemplate'])
                if target_request.get('InputTransformer').get('InputPathsMap', None):
                    target_request['InputTransformer']['InputPathsMap'] = target['input_transformer']['input_paths_map']
            targets_request.append(target_request)
        return targets_request