from openstack import resource
class MetadefSchema(resource.Resource):
    base_path = '/schemas/metadefs'
    allow_fetch = True
    additional_properties = resource.Body('additionalProperties', type=bool)
    definitions = resource.Body('definitions', type=dict)
    required = resource.Body('required', type=list)
    properties = resource.Body('properties', type=dict)