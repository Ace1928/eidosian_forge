import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.example import ResponseExampleDocumenter
from botocore.docs.method import (
from botocore.docs.params import ResponseParamsDocumenter
from botocore.docs.sharedexample import document_shared_examples
from botocore.docs.utils import DocumentedShape, get_official_service_name
class ClientContextParamsDocumenter:
    _CONFIG_GUIDE_LINK = 'https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html'
    OMITTED_CONTEXT_PARAMS = {'s3': ('Accelerate', 'DisableMultiRegionAccessPoints', 'ForcePathStyle', 'UseArnRegion'), 's3control': ('UseArnRegion',)}

    def __init__(self, service_name, context_params):
        self._service_name = service_name
        self._context_params = context_params

    def document_context_params(self, section):
        self._add_title(section)
        self._add_overview(section)
        self._add_context_params_list(section)

    def _add_title(self, section):
        section.style.h2('Client Context Parameters')

    def _add_overview(self, section):
        section.style.new_line()
        section.write('Client context parameters are configurable on a client instance via the ``client_context_params`` parameter in the ``Config`` object. For more detailed instructions and examples on the exact usage of context params see the ')
        section.style.external_link(title='configuration guide', link=self._CONFIG_GUIDE_LINK)
        section.write('.')
        section.style.new_line()

    def _add_context_params_list(self, section):
        section.style.new_line()
        sn = f'``{self._service_name}``'
        section.writeln(f'The available {sn} client context params are:')
        for param in self._context_params:
            section.style.new_line()
            name = f'``{xform_name(param.name)}``'
            section.write(f'* {name} ({param.type}) - {param.documentation}')