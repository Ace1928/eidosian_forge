from openstack import resource
class SfcFlowClassifier(resource.Resource):
    resource_key = 'flow_classifier'
    resources_key = 'flow_classifiers'
    base_path = '/sfc/flow_classifiers'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('description', 'name', 'project_id', 'tenant_id', 'ethertype', 'protocol', 'source_port_range_min', 'source_port_range_max', 'destination_port_range_min', 'destination_port_range_max', 'logical_source_port', 'logical_destination_port')
    description = resource.Body('description')
    name = resource.Body('name')
    ethertype = resource.Body('ethertype')
    protocol = resource.Body('protocol')
    source_port_range_min = resource.Body('source_port_range_min', type=int)
    source_port_range_max = resource.Body('source_port_range_max', type=int)
    destination_port_range_min = resource.Body('destination_port_range_min', type=int)
    destination_port_range_max = resource.Body('destination_port_range_max', type=int)
    source_ip_prefix = resource.Body('source_ip_prefix')
    destination_ip_prefix = resource.Body('destination_ip_prefix')
    logical_source_port = resource.Body('logical_source_port')
    logical_destination_port = resource.Body('logical_destination_port')
    l7_parameters = resource.Body('l7_parameters', type=dict)
    summary = resource.Computed('summary', default='')
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)