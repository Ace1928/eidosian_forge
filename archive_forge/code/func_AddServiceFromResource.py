import collections
import logging
import re
import textwrap
from apitools.base.py import base_api
from apitools.gen import util
def AddServiceFromResource(self, service_name, methods):
    """Add a new service named service_name with the given methods."""
    service_name = self.__names.CleanName(service_name)
    method_descriptions = methods.get('methods', {})
    method_info_map = collections.OrderedDict()
    items = sorted(method_descriptions.items())
    for method_name, method_description in items:
        method_name = self.__names.MethodName(method_name)
        body_type = method_description.get('request')
        if body_type is None:
            request_type = None
        else:
            request_type = self.__GetRequestType(body_type)
        if self.__NeedRequestType(method_description, request_type):
            request = self.__CreateRequestType(method_description, body_type=body_type)
            request_field = self.__GetRequestField(method_description, body_type)
        else:
            request = request_type
            request_field = base_api.REQUEST_IS_BODY
        if 'response' in method_description:
            response = method_description['response']['$ref']
        else:
            response = self.__CreateVoidResponseType(method_description)
        method_info_map[method_name] = self.__ComputeMethodInfo(method_description, request, response, request_field)
    nested_services = methods.get('resources', {})
    services = sorted(nested_services.items())
    for subservice_name, submethods in services:
        new_service_name = '%s_%s' % (service_name, subservice_name)
        self.AddServiceFromResource(new_service_name, submethods)
    self.__RegisterService(service_name, method_info_map)