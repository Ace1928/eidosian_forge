import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.example import ResponseExampleDocumenter
from botocore.docs.method import (
from botocore.docs.params import ResponseParamsDocumenter
from botocore.docs.sharedexample import document_shared_examples
from botocore.docs.utils import DocumentedShape, get_official_service_name
def _add_response_example(self, section, shape):
    example_section = section.add_new_section('syntax')
    example_section.style.new_line()
    example_section.style.bold('Syntax')
    example_section.style.new_paragraph()
    documenter = ResponseExampleDocumenter(service_name=self._service_name, operation_name=None, event_emitter=self._client.meta.events)
    documenter.document_example(example_section, shape, include=[self._GENERIC_ERROR_SHAPE])