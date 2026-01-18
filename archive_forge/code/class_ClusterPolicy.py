from openstack import resource
class ClusterPolicy(resource.Resource):
    resource_key = 'cluster_policy'
    resources_key = 'cluster_policies'
    base_path = '/clusters/%(cluster_id)s/policies'
    allow_list = True
    allow_fetch = True
    _query_mapping = resource.QueryParameters('sort', 'policy_name', 'policy_type', is_enabled='enabled')
    policy_id = resource.Body('policy_id', alternate_id=True)
    policy_name = resource.Body('policy_name')
    cluster_id = resource.URI('cluster_id')
    cluster_name = resource.Body('cluster_name')
    policy_type = resource.Body('policy_type')
    is_enabled = resource.Body('enabled', type=bool)
    data = resource.Body('data', type=dict)