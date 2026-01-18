import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.example import ResponseExampleDocumenter
from botocore.docs.method import (
from botocore.docs.params import ResponseParamsDocumenter
from botocore.docs.sharedexample import document_shared_examples
from botocore.docs.utils import DocumentedShape, get_official_service_name
def _add_client_method(self, section, method_name, method):
    breadcrumb_section = section.add_new_section('breadcrumb')
    breadcrumb_section.style.ref(self._client_class_name, f'../../{self._service_name}')
    breadcrumb_section.write(f' / Client / {method_name}')
    section.add_title_section(method_name)
    method_section = section.add_new_section(method_name, context={'qualifier': f'{self._client_class_name}.Client.'})
    if self._is_custom_method(method_name):
        self._add_custom_method(method_section, method_name, method)
    else:
        self._add_model_driven_method(method_section, method_name)