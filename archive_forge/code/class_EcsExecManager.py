from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class EcsExecManager:
    """Handles ECS Tasks"""

    def __init__(self, module):
        self.module = module
        self.ecs = module.client('ecs')
        self.ec2 = module.client('ec2')

    def format_network_configuration(self, network_config):
        result = dict()
        if 'subnets' in network_config:
            result['subnets'] = network_config['subnets']
        else:
            self.module.fail_json(msg='Network configuration must include subnets')
        if 'security_groups' in network_config:
            groups = network_config['security_groups']
            if any((not sg.startswith('sg-') for sg in groups)):
                try:
                    vpc_id = self.ec2.describe_subnets(SubnetIds=[result['subnets'][0]])['Subnets'][0]['VpcId']
                    groups = get_ec2_security_group_ids_from_names(groups, self.ec2, vpc_id)
                except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                    self.module.fail_json_aws(e, msg="Couldn't look up security groups")
            result['securityGroups'] = groups
        if 'assign_public_ip' in network_config:
            if network_config['assign_public_ip'] is True:
                result['assignPublicIp'] = 'ENABLED'
            else:
                result['assignPublicIp'] = 'DISABLED'
        return dict(awsvpcConfiguration=result)

    def list_tasks(self, cluster_name, service_name, status):
        response = self.ecs.list_tasks(cluster=cluster_name, family=service_name, desiredStatus=status)
        if len(response['taskArns']) > 0:
            for c in response['taskArns']:
                if c.endswith(service_name):
                    return c
        return None

    def run_task(self, cluster, task_definition, overrides, count, startedBy, launch_type, tags):
        if overrides is None:
            overrides = dict()
        params = dict(cluster=cluster, taskDefinition=task_definition, overrides=overrides, count=count, startedBy=startedBy)
        if self.module.params['network_configuration']:
            params['networkConfiguration'] = self.format_network_configuration(self.module.params['network_configuration'])
        if launch_type:
            params['launchType'] = launch_type
        if tags:
            params['tags'] = ansible_dict_to_boto3_tag_list(tags, 'key', 'value')
        try:
            response = self.ecs.run_task(**params)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.module.fail_json_aws(e, msg="Couldn't run task")
        return response['tasks']

    def start_task(self, cluster, task_definition, overrides, container_instances, startedBy, tags):
        args = dict()
        if cluster:
            args['cluster'] = cluster
        if task_definition:
            args['taskDefinition'] = task_definition
        if overrides:
            args['overrides'] = overrides
        if container_instances:
            args['containerInstances'] = container_instances
        if startedBy:
            args['startedBy'] = startedBy
        if self.module.params['network_configuration']:
            args['networkConfiguration'] = self.format_network_configuration(self.module.params['network_configuration'])
        if tags:
            args['tags'] = ansible_dict_to_boto3_tag_list(tags, 'key', 'value')
        try:
            response = self.ecs.start_task(**args)
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self.module.fail_json_aws(e, msg="Couldn't start task")
        return response['tasks']

    def stop_task(self, cluster, task):
        response = self.ecs.stop_task(cluster=cluster, task=task)
        return response['task']

    def ecs_task_long_format_enabled(self):
        account_support = self.ecs.list_account_settings(name='taskLongArnFormat', effectiveSettings=True)
        return account_support['settings'][0]['value'] == 'enabled'