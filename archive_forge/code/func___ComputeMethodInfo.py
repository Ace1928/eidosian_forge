import collections
import logging
import re
import textwrap
from apitools.base.py import base_api
from apitools.gen import util
def __ComputeMethodInfo(self, method_description, request, response, request_field):
    """Compute the base_api.ApiMethodInfo for this method."""
    relative_path = self.__names.NormalizeRelativePath(''.join((self.__client_info.base_path, method_description['path'])))
    method_id = method_description['id']
    ordered_params = []
    for param_name in method_description.get('parameterOrder', []):
        param_info = method_description['parameters'][param_name]
        if param_info.get('required', False):
            ordered_params.append(param_name)
    method_info = base_api.ApiMethodInfo(relative_path=relative_path, method_id=method_id, http_method=method_description['httpMethod'], description=util.CleanDescription(method_description.get('description', '')), query_params=[], path_params=[], ordered_params=ordered_params, request_type_name=self.__names.ClassName(request), response_type_name=self.__names.ClassName(response), request_field=request_field)
    flat_path = method_description.get('flatPath', None)
    if flat_path is not None:
        flat_path = self.__names.NormalizeRelativePath(self.__client_info.base_path + flat_path)
        if flat_path != relative_path:
            method_info.flat_path = flat_path
    if method_description.get('supportsMediaUpload', False):
        method_info.upload_config = self.__ComputeUploadConfig(method_description.get('mediaUpload'), method_id)
    method_info.supports_download = method_description.get('supportsMediaDownload', False)
    self.__all_scopes.update(method_description.get('scopes', ()))
    for param, desc in method_description.get('parameters', {}).items():
        param = self.__names.CleanName(param)
        location = desc['location']
        if location == 'query':
            method_info.query_params.append(param)
        elif location == 'path':
            method_info.path_params.append(param)
        else:
            raise ValueError('Unknown parameter location %s for parameter %s' % (location, param))
    method_info.path_params.sort()
    method_info.query_params.sort()
    return method_info