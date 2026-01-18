import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.example import ResponseExampleDocumenter
from botocore.docs.method import (
from botocore.docs.params import ResponseParamsDocumenter
from botocore.docs.sharedexample import document_shared_examples
from botocore.docs.utils import DocumentedShape, get_official_service_name
def _add_response_attr(self, section, shape):
    response_section = section.add_new_section('response')
    response_section.style.start_sphinx_py_attr('response')
    self._add_response_attr_description(response_section)
    self._add_response_example(response_section, shape)
    self._add_response_params(response_section, shape)
    response_section.style.end_sphinx_py_attr()