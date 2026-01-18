import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class EcsClusterManager:
    """Handles ECS Clusters"""

    def __init__(self, module):
        self.module = module
        try:
            self.ecs = module.client('ecs')
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg='Failed to connect to AWS')

    def find_in_array(self, array_of_clusters, cluster_name, field_name='clusterArn'):
        for c in array_of_clusters:
            if c[field_name].endswith(cluster_name):
                return c
        return None

    def describe_cluster(self, cluster_name):
        response = self.ecs.describe_clusters(clusters=[cluster_name])
        if len(response['failures']) > 0:
            c = self.find_in_array(response['failures'], cluster_name, 'arn')
            if c and c['reason'] == 'MISSING':
                return None
        if len(response['clusters']) > 0:
            c = self.find_in_array(response['clusters'], cluster_name)
            if c:
                return c
        raise Exception(f'Unknown problem describing cluster {cluster_name}.')

    def create_cluster(self, cluster_name, capacity_providers, capacity_provider_strategy):
        params = dict(clusterName=cluster_name)
        if capacity_providers:
            params['capacityProviders'] = snake_dict_to_camel_dict(capacity_providers)
        if capacity_provider_strategy:
            params['defaultCapacityProviderStrategy'] = snake_dict_to_camel_dict(capacity_provider_strategy)
        response = self.ecs.create_cluster(**params)
        return response['cluster']

    def update_cluster(self, cluster_name, capacity_providers, capacity_provider_strategy):
        params = dict(cluster=cluster_name)
        if capacity_providers:
            params['capacityProviders'] = snake_dict_to_camel_dict(capacity_providers)
        else:
            params['capacityProviders'] = []
        if capacity_provider_strategy:
            params['defaultCapacityProviderStrategy'] = snake_dict_to_camel_dict(capacity_provider_strategy)
        else:
            params['defaultCapacityProviderStrategy'] = []
        response = self.ecs.put_cluster_capacity_providers(**params)
        return response['cluster']

    def delete_cluster(self, clusterName):
        return self.ecs.delete_cluster(cluster=clusterName)