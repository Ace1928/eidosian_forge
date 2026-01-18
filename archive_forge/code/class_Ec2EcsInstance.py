from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class Ec2EcsInstance(object):
    """Handle ECS Cluster Remote Operations"""

    def __init__(self, module, cluster, ec2_id):
        self.module = module
        self.cluster = cluster
        self.ec2_id = ec2_id
        try:
            self.ecs = module.client('ecs')
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg='Failed to connect to AWS')
        self.ecs_arn = self._get_ecs_arn()

    def _get_ecs_arn(self):
        try:
            ecs_instances_arns = self.ecs.list_container_instances(cluster=self.cluster)['containerInstanceArns']
            ec2_instances = self.ecs.describe_container_instances(cluster=self.cluster, containerInstances=ecs_instances_arns)['containerInstances']
        except (ClientError, EndpointConnectionError) as e:
            self.module.fail_json(msg=f"Can't connect to the cluster - {str(e)}")
        try:
            ecs_arn = next((inst for inst in ec2_instances if inst['ec2InstanceId'] == self.ec2_id))['containerInstanceArn']
        except StopIteration:
            self.module.fail_json(msg=f'EC2 instance Id not found in ECS cluster - {str(self.cluster)}')
        return ecs_arn

    def attrs_put(self, attrs):
        """Puts attributes on ECS container instance"""
        try:
            self.ecs.put_attributes(cluster=self.cluster, attributes=attrs.get_for_ecs_arn(self.ecs_arn))
        except ClientError as e:
            self.module.fail_json(msg=str(e))

    def attrs_delete(self, attrs):
        """Deletes attributes from ECS container instance."""
        try:
            self.ecs.delete_attributes(cluster=self.cluster, attributes=attrs.get_for_ecs_arn(self.ecs_arn, skip_value=True))
        except ClientError as e:
            self.module.fail_json(msg=str(e))

    def attrs_get_by_name(self, attrs):
        """
        Returns EcsAttributes object containing attributes from ECS container instance with names
        matching to attrs.attributes (EcsAttributes Object).
        """
        attr_objs = [{'targetType': 'container-instance', 'attributeName': attr['name']} for attr in attrs]
        try:
            matched_ecs_targets = [attr_found for attr_obj in attr_objs for attr_found in self.ecs.list_attributes(cluster=self.cluster, **attr_obj)['attributes']]
        except ClientError as e:
            self.module.fail_json(msg=f"Can't connect to the cluster - {str(e)}")
        matched_objs = [target for target in matched_ecs_targets if target['targetId'] == self.ecs_arn]
        results = [{'name': match['name'], 'value': match.get('value', None)} for match in matched_objs]
        return EcsAttributes(self.module, results)