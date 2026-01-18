import collections
import logging
import re
import textwrap
from apitools.base.py import base_api
from apitools.gen import util
def __WriteSingleService(self, printer, name, method_info_map, client_class_name):
    printer()
    class_name = self.__GetServiceClassName(name)
    printer('class %s(base_api.BaseApiService):', class_name)
    with printer.Indent():
        printer('"""Service class for the %s resource."""', name)
        printer()
        printer('_NAME = %s', repr(name))
        printer()
        printer('def __init__(self, client):')
        with printer.Indent():
            printer('super(%s.%s, self).__init__(client)', client_class_name, class_name)
            printer('self._upload_configs = {')
            with printer.Indent(indent='    '):
                for method_name, method_info in method_info_map.items():
                    upload_config = method_info.upload_config
                    if upload_config is not None:
                        printer("'%s': base_api.ApiUploadInfo(", method_name)
                        with printer.Indent(indent='    '):
                            attrs = sorted((x.name for x in upload_config.all_fields()))
                            for attr in attrs:
                                printer('%s=%r,', attr, getattr(upload_config, attr))
                        printer('),')
                printer('}')
        for method_name, method_info in method_info_map.items():
            printer()
            params = ['self', 'request', 'global_params=None']
            if method_info.upload_config:
                params.append('upload=None')
            if method_info.supports_download:
                params.append('download=None')
            printer('def %s(%s):', method_name, ', '.join(params))
            with printer.Indent():
                self.__PrintDocstring(printer, method_info, method_name, name)
                printer("config = self.GetMethodConfig('%s')", method_name)
                upload_config = method_info.upload_config
                if upload_config is not None:
                    printer("upload_config = self.GetUploadConfig('%s')", method_name)
                arg_lines = ['config, request, global_params=global_params']
                if method_info.upload_config:
                    arg_lines.append('upload=upload, upload_config=upload_config')
                if method_info.supports_download:
                    arg_lines.append('download=download')
                printer('return self._RunMethod(')
                with printer.Indent(indent='    '):
                    for line in arg_lines[:-1]:
                        printer('%s,', line)
                    printer('%s)', arg_lines[-1])
            printer()
            printer('{0}.method_config = lambda: base_api.ApiMethodInfo('.format(method_name))
            with printer.Indent(indent='    '):
                method_info = method_info_map[method_name]
                attrs = sorted((x.name for x in method_info.all_fields()))
                for attr in attrs:
                    if attr in ('upload_config', 'description'):
                        continue
                    value = getattr(method_info, attr)
                    if value is not None:
                        printer('%s=%r,', attr, value)
            printer(')')