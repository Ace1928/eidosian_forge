from openstack import resource
class FederationProtocol(resource.Resource):
    resource_key = 'protocol'
    resources_key = 'protocols'
    base_path = '/OS-FEDERATION/identity_providers/%(idp_id)s/protocols'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    create_exclude_id_from_body = True
    create_method = 'PUT'
    commit_method = 'PATCH'
    _query_mapping = resource.QueryParameters('id')
    name = resource.Body('id')
    idp_id = resource.URI('idp_id')
    mapping_id = resource.Body('mapping_id')