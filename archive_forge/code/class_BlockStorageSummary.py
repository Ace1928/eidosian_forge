from openstack import resource
class BlockStorageSummary(resource.Resource):
    base_path = '/volumes/summary'
    allow_fetch = True
    total_size = resource.Body('total_size')
    total_count = resource.Body('total_count')
    metadata = resource.Body('metadata')
    _max_microversion = '3.36'