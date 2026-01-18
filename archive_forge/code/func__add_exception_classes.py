import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.example import ResponseExampleDocumenter
from botocore.docs.method import (
from botocore.docs.params import ResponseParamsDocumenter
from botocore.docs.sharedexample import document_shared_examples
from botocore.docs.utils import DocumentedShape, get_official_service_name
def _add_exception_classes(self):
    for shape in self._client.meta.service_model.error_shapes:
        exception_doc_structure = DocumentStructure(shape.name, target='html')
        self._add_exception_class(exception_doc_structure, shape)
        exception_dir_path = os.path.join(self._root_docs_path, self._service_name, 'client', 'exceptions')
        exception_doc_structure.write_to_file(exception_dir_path, shape.name)