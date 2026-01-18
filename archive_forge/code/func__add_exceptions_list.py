import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.example import ResponseExampleDocumenter
from botocore.docs.method import (
from botocore.docs.params import ResponseParamsDocumenter
from botocore.docs.sharedexample import document_shared_examples
from botocore.docs.utils import DocumentedShape, get_official_service_name
def _add_exceptions_list(self, section):
    error_shapes = self._client.meta.service_model.error_shapes
    if not error_shapes:
        section.style.new_line()
        section.write('This client has no modeled exception classes.')
        section.style.new_line()
        return
    section.style.new_line()
    section.writeln('The available client exceptions are:')
    section.style.toctree()
    for shape in error_shapes:
        section.style.tocitem(f'{self._service_name}/client/exceptions/{shape.name}')