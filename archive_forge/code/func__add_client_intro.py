import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.example import ResponseExampleDocumenter
from botocore.docs.method import (
from botocore.docs.params import ResponseParamsDocumenter
from botocore.docs.sharedexample import document_shared_examples
from botocore.docs.utils import DocumentedShape, get_official_service_name
def _add_client_intro(self, section, client_methods):
    section = section.add_new_section('intro')
    official_service_name = get_official_service_name(self._client.meta.service_model)
    section.write(f'A low-level client representing {official_service_name}')
    section.style.new_line()
    section.include_doc_string(self._client.meta.service_model.documentation)
    self._add_client_creation_example(section)
    section.style.dedent()
    section.style.new_paragraph()
    section.writeln('These are the available methods:')
    section.style.toctree()
    for method_name in sorted(client_methods):
        section.style.tocitem(f'{self._service_name}/client/{method_name}')