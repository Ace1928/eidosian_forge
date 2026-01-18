from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel, _camel_to_snake
import re
class AzureRMIoTHub(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), state=dict(type='str', default='present', choices=['present', 'absent']), location=dict(type='str'), sku=dict(type='str', choices=['b1', 'b2', 'b3', 'f1', 's1', 's2', 's3']), unit=dict(type='int'), event_endpoint=dict(type='dict', options=event_endpoint_spec), enable_file_upload_notifications=dict(type='bool'), ip_filters=dict(type='list', elements='dict', options=ip_filter_spec), routing_endpoints=dict(type='list', elements='dict', options=routing_endpoints_spec), routes=dict(type='list', elements='dict', options=routes_spec))
        self.results = dict(changed=False, id=None)
        self.resource_group = None
        self.name = None
        self.state = None
        self.location = None
        self.sku = None
        self.unit = None
        self.event_endpoint = None
        self.tags = None
        self.enable_file_upload_notifications = None
        self.ip_filters = None
        self.routing_endpoints = None
        self.routes = None
        super(AzureRMIoTHub, self).__init__(self.module_arg_spec, supports_check_mode=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        changed = False
        if not self.location:
            resource_group = self.get_resource_group(self.resource_group)
            self.location = resource_group.location
        self.sku = str.capitalize(self.sku) if self.sku else None
        iothub = self.get_hub()
        if self.state == 'present':
            if not iothub:
                changed = True
                self.sku = self.sku or 'S1'
                self.unit = self.unit or 1
                self.event_endpoint = self.event_endpoint or {}
                self.event_endpoint['partition_count'] = self.event_endpoint.get('partition_count') or 2
                self.event_endpoint['retention_time_in_days'] = self.event_endpoint.get('retention_time_in_days') or 1
                event_hub_properties = dict()
                event_hub_properties['events'] = self.IoThub_models.EventHubProperties(**self.event_endpoint)
                iothub_property = self.IoThub_models.IotHubProperties(event_hub_endpoints=event_hub_properties)
                if self.enable_file_upload_notifications:
                    iothub_property.enable_file_upload_notifications = self.enable_file_upload_notifications
                if self.ip_filters:
                    iothub_property.ip_filter_rules = self.construct_ip_filters()
                routing_endpoints = None
                routes = None
                if self.routing_endpoints:
                    routing_endpoints = self.construct_routing_endpoint(self.routing_endpoints)
                if self.routes:
                    routes = [self.construct_route(x) for x in self.routes]
                if routes or routing_endpoints:
                    routing_property = self.IoThub_models.RoutingProperties(endpoints=routing_endpoints, routes=routes)
                    iothub_property.routing = routing_property
                iothub = self.IoThub_models.IotHubDescription(location=self.location, sku=self.IoThub_models.IotHubSkuInfo(name=self.sku, capacity=self.unit), properties=iothub_property, tags=self.tags)
                if not self.check_mode:
                    iothub = self.create_or_update_hub(iothub)
            else:
                original_sku = iothub.sku
                if self.sku and self.sku != original_sku.name:
                    self.log('SKU changed')
                    iothub.sku.name = self.sku
                    changed = True
                if self.unit and self.unit != original_sku.capacity:
                    self.log('Unit count changed')
                    iothub.sku.capacity = self.unit
                    changed = True
                event_hub = iothub.properties.event_hub_endpoints or dict()
                if self.event_endpoint:
                    item = self.event_endpoint
                    original_item = event_hub.get('events')
                    if not original_item:
                        changed = True
                        event_hub['events'] = self.IoThub_models.EventHubProperties(partition_count=item.get('partition_count') or 2, retention_time_in_days=item.get('retention_time_in_days') or 1)
                    elif item.get('partition_count') and original_item.partition_count != item['partition_count']:
                        changed = True
                        original_item.partition_count = item['partition_count']
                    elif item.get('retention_time_in_days') and original_item.retention_time_in_days != item['retention_time_in_days']:
                        changed = True
                        original_item.retention_time_in_days = item['retention_time_in_days']
                original_endpoints = iothub.properties.routing.endpoints
                endpoint_changed = False
                if self.routing_endpoints:
                    total_length = 0
                    for item in routing_endpoints_resource_type_mapping.values():
                        attribute = item['attribute']
                        array = getattr(original_endpoints, attribute)
                        total_length += len(array or [])
                    if total_length != len(self.routing_endpoints):
                        endpoint_changed = True
                    else:
                        for item in self.routing_endpoints:
                            if not self.lookup_endpoint(item, original_endpoints):
                                endpoint_changed = True
                                break
                    if endpoint_changed:
                        iothub.properties.routing.endpoints = self.construct_routing_endpoint(self.routing_endpoints)
                        changed = True
                original_routes = iothub.properties.routing.routes
                routes_changed = False
                if self.routes:
                    if len(self.routes) != len(original_routes or []):
                        routes_changed = True
                    else:
                        for item in self.routes:
                            if not self.lookup_route(item, original_routes):
                                routes_changed = True
                                break
                    if routes_changed:
                        changed = True
                        iothub.properties.routing.routes = [self.construct_route(x) for x in self.routes]
                ip_filter_changed = False
                original_ip_filter = iothub.properties.ip_filter_rules
                if self.ip_filters:
                    if len(self.ip_filters) != len(original_ip_filter or []):
                        ip_filter_changed = True
                    else:
                        for item in self.ip_filters:
                            if not self.lookup_ip_filter(item, original_ip_filter):
                                ip_filter_changed = True
                                break
                    if ip_filter_changed:
                        changed = True
                        iothub.properties.ip_filter_rules = self.construct_ip_filters()
                tag_changed, updated_tags = self.update_tags(iothub.tags)
                iothub.tags = updated_tags
                if changed and (not self.check_mode):
                    iothub = self.create_or_update_hub(iothub)
                if not changed and tag_changed:
                    changed = True
                    if not self.check_mode:
                        iothub = self.update_instance_tags(updated_tags)
            self.results = self.to_dict(iothub)
        elif iothub:
            changed = True
            if not self.check_mode:
                self.delete_hub()
        self.results['changed'] = changed
        return self.results

    def lookup_ip_filter(self, target, ip_filters):
        if not ip_filters or len(ip_filters) == 0:
            return False
        for item in ip_filters:
            if item.filter_name == target['name']:
                if item.ip_mask != target['ip_mask']:
                    return False
                if item.action.lower() != target['action']:
                    return False
                return True
        return False

    def lookup_route(self, target, routes):
        if not routes or len(routes) == 0:
            return False
        for item in routes:
            if item.name == target['name']:
                if target['source'] != _camel_to_snake(item.source):
                    return False
                if target['enabled'] != item.is_enabled:
                    return False
                if target['endpoint_name'] != item.endpoint_names[0]:
                    return False
                if target.get('condition') and target['condition'] != item.condition:
                    return False
                return True
        return False

    def lookup_endpoint(self, target, routing_endpoints):
        resource_type = target['resource_type']
        attribute = routing_endpoints_resource_type_mapping[resource_type]['attribute']
        endpoints = getattr(routing_endpoints, attribute)
        if not endpoints or len(endpoints) == 0:
            return False
        for item in endpoints:
            if item.name == target['name']:
                if target.get('resource_group') and target['resource_group'] != (item.resource_group or self.resource_group):
                    return False
                if target.get('subscription_id') and target['subscription_id'] != (item.subscription_id or self.subscription_id):
                    return False
                connection_string_regex = item.connection_string.replace('****', '.*')
                connection_string_regex = re.sub(':\\d+/;', '/;', connection_string_regex)
                if not re.search(connection_string_regex, target['connection_string']):
                    return False
                if resource_type == 'storage':
                    if target.get('container') and item.container_name != target['container']:
                        return False
                    if target.get('encoding') and item.encoding != target['encoding']:
                        return False
                return True
        return False

    def construct_ip_filters(self):
        return [self.IoThub_models.IpFilterRule(filter_name=x['name'], action=self.IoThub_models.IpFilterActionType[x['action']], ip_mask=x['ip_mask']) for x in self.ip_filters]

    def construct_routing_endpoint(self, routing_endpoints):
        if not routing_endpoints or len(routing_endpoints) == 0:
            return None
        result = self.IoThub_models.RoutingEndpoints()
        for endpoint in routing_endpoints:
            resource_type_property = routing_endpoints_resource_type_mapping.get(endpoint['resource_type'])
            resource_type = getattr(self.IoThub_models, resource_type_property['model'])
            array = getattr(result, resource_type_property['attribute']) or []
            array.append(resource_type(**endpoint))
            setattr(result, resource_type_property['attribute'], array)
        return result

    def construct_route(self, route):
        if not route:
            return None
        return self.IoThub_models.RouteProperties(name=route['name'], source=_snake_to_camel(snake=route['source'], capitalize_first=True), is_enabled=route['enabled'], endpoint_names=[route['endpoint_name']], condition=route.get('condition'))

    def get_hub(self):
        try:
            return self.IoThub_client.iot_hub_resource.get(self.resource_group, self.name)
        except Exception:
            return None

    def create_or_update_hub(self, hub):
        try:
            poller = self.IoThub_client.iot_hub_resource.begin_create_or_update(self.resource_group, self.name, hub, if_match=hub.etag)
            return self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error creating or updating IoT Hub {0}: {1}'.format(self.name, exc.message or str(exc)))

    def update_instance_tags(self, tags):
        try:
            poller = self.IoThub_client.iot_hub_resource.begin_update(self.resource_group, self.name, tags=tags)
            return self.get_poller_result(poller)
        except Exception as exc:
            self.fail("Error updating IoT Hub {0}'s tag: {1}".format(self.name, exc.message or str(exc)))

    def delete_hub(self):
        try:
            self.IoThub_client.iot_hub_resource.begin_delete(self.resource_group, self.name)
            return True
        except Exception as exc:
            self.fail('Error deleting IoT Hub {0}: {1}'.format(self.name, exc.message or str(exc)))
            return False

    def route_to_dict(self, route):
        return dict(name=route.name, source=_camel_to_snake(route.source), endpoint_name=route.endpoint_names[0], enabled=route.is_enabled, condition=route.condition)

    def instance_dict_to_dict(self, instance_dict):
        result = dict()
        if not instance_dict:
            return result
        for key in instance_dict.keys():
            result[key] = instance_dict[key].as_dict()
        return result

    def to_dict(self, hub):
        result = dict()
        properties = hub.properties
        result['id'] = hub.id
        result['name'] = hub.name
        result['resource_group'] = self.resource_group
        result['location'] = hub.location
        result['tags'] = hub.tags
        result['unit'] = hub.sku.capacity
        result['sku'] = hub.sku.name.lower()
        result['cloud_to_device'] = dict(max_delivery_count=properties.cloud_to_device.feedback.max_delivery_count, ttl_as_iso8601=str(properties.cloud_to_device.feedback.ttl_as_iso8601)) if properties.cloud_to_device else dict()
        result['enable_file_upload_notifications'] = properties.enable_file_upload_notifications
        result['event_endpoint'] = properties.event_hub_endpoints.get('events').as_dict() if properties.event_hub_endpoints.get('events') else None
        result['host_name'] = properties.host_name
        result['ip_filters'] = [x.as_dict() for x in properties.ip_filter_rules]
        if properties.routing:
            result['routing_endpoints'] = properties.routing.endpoints.as_dict()
            result['routes'] = [self.route_to_dict(x) for x in properties.routing.routes]
            result['fallback_route'] = self.route_to_dict(properties.routing.fallback_route)
        result['status'] = properties.state
        result['storage_endpoints'] = self.instance_dict_to_dict(properties.storage_endpoints)
        return result