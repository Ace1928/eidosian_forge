from openstack import resource
class ProviderFlavorCapabilities(resource.Resource):
    resources_key = 'flavor_capabilities'
    base_path = '/lbaas/providers/%(provider)s/flavor_capabilities'
    allow_create = False
    allow_fetch = False
    allow_commit = False
    allow_delete = False
    allow_list = True
    _query_mapping = resource.QueryParameters('description', 'name')
    provider = resource.URI('provider')
    name = resource.Body('name')
    description = resource.Body('description')