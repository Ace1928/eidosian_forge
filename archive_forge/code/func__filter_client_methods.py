import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.example import ResponseExampleDocumenter
from botocore.docs.method import (
from botocore.docs.params import ResponseParamsDocumenter
from botocore.docs.sharedexample import document_shared_examples
from botocore.docs.utils import DocumentedShape, get_official_service_name
def _filter_client_methods(self, client_methods):
    filtered_methods = {}
    for method_name, method in client_methods.items():
        include = self._filter_client_method(method=method, method_name=method_name, service_name=self._service_name)
        if include:
            filtered_methods[method_name] = method
    return filtered_methods