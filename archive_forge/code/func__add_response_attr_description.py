import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.example import ResponseExampleDocumenter
from botocore.docs.method import (
from botocore.docs.params import ResponseParamsDocumenter
from botocore.docs.sharedexample import document_shared_examples
from botocore.docs.utils import DocumentedShape, get_official_service_name
def _add_response_attr_description(self, section):
    section.style.new_line()
    section.include_doc_string('The parsed error response. All exceptions have a top level ``Error`` key that provides normalized access to common exception atrributes. All other keys are specific to this service or exception class.')
    section.style.new_line()