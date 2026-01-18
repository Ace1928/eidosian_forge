import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.example import ResponseExampleDocumenter
from botocore.docs.method import (
from botocore.docs.params import ResponseParamsDocumenter
from botocore.docs.sharedexample import document_shared_examples
from botocore.docs.utils import DocumentedShape, get_official_service_name
def _add_client_methods(self, client_methods):
    for method_name in sorted(client_methods):
        method_doc_structure = DocumentStructure(method_name, target='html')
        self._add_client_method(method_doc_structure, method_name, client_methods[method_name])
        client_dir_path = os.path.join(self._root_docs_path, self._service_name, 'client')
        method_doc_structure.write_to_file(client_dir_path, method_name)