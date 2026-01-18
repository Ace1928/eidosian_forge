from openstack import resource
class MeteringLabelRule(resource.Resource):
    resource_key = 'metering_label_rule'
    resources_key = 'metering_label_rules'
    base_path = '/metering/metering-label-rules'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('direction', 'metering_label_id', 'remote_ip_prefix', 'source_ip_prefix', 'destination_ip_prefix', 'project_id', 'sort_key', 'sort_dir')
    direction = resource.Body('direction')
    is_excluded = resource.Body('excluded', type=bool)
    metering_label_id = resource.Body('metering_label_id')
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)
    remote_ip_prefix = resource.Body('remote_ip_prefix', deprecated=True, deprecation_reason="The use of 'remote_ip_prefix' in metering label rules is deprecated and will be removed in future releases. One should use instead, the 'source_ip_prefix' and/or 'destination_ip_prefix' parameters. For more details, you can check the spec: https://review.opendev.org/#/c/744702/.")
    source_ip_prefix = resource.Body('source_ip_prefix')
    destination_ip_prefix = resource.Body('destination_ip_prefix')