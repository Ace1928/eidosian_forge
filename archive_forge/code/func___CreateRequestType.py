import collections
import logging
import re
import textwrap
from apitools.base.py import base_api
from apitools.gen import util
def __CreateRequestType(self, method_description, body_type=None):
    """Create a request type for this method."""
    schema = {}
    schema['id'] = self.__names.ClassName('%sRequest' % (self.__names.ClassName(method_description['id'], separator='.'),))
    schema['type'] = 'object'
    schema['properties'] = collections.OrderedDict()
    if 'parameterOrder' not in method_description:
        ordered_parameters = list(method_description.get('parameters', []))
    else:
        ordered_parameters = method_description['parameterOrder'][:]
        for k in method_description['parameters']:
            if k not in ordered_parameters:
                ordered_parameters.append(k)
    for parameter_name in ordered_parameters:
        field_name = self.__names.CleanName(parameter_name)
        field = dict(method_description['parameters'][parameter_name])
        if 'type' not in field:
            raise ValueError('No type found in parameter %s' % field)
        schema['properties'][field_name] = field
    if body_type is not None:
        body_field_name = self.__GetRequestField(method_description, body_type)
        if body_field_name in schema['properties']:
            raise ValueError('Failed to normalize request resource name')
        if 'description' not in body_type:
            body_type['description'] = 'A %s resource to be passed as the request body.' % (self.__GetRequestType(body_type),)
        schema['properties'][body_field_name] = body_type
    self.__message_registry.AddDescriptorFromSchema(schema['id'], schema)
    return schema['id']