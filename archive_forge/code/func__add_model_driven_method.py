import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.example import ResponseExampleDocumenter
from botocore.docs.method import (
from botocore.docs.params import ResponseParamsDocumenter
from botocore.docs.sharedexample import document_shared_examples
from botocore.docs.utils import DocumentedShape, get_official_service_name
def _add_model_driven_method(self, section, method_name):
    service_model = self._client.meta.service_model
    operation_name = self._client.meta.method_to_api_mapping[method_name]
    operation_model = service_model.operation_model(operation_name)
    example_prefix = 'response = client.%s' % method_name
    full_method_name = f'{section.context.get('qualifier', '')}{method_name}'
    document_model_driven_method(section, full_method_name, operation_model, event_emitter=self._client.meta.events, method_description=operation_model.documentation, example_prefix=example_prefix)
    if operation_model.error_shapes:
        self._add_method_exceptions_list(section, operation_model)
    shared_examples = self._shared_examples.get(operation_name)
    if shared_examples:
        document_shared_examples(section, operation_model, example_prefix, shared_examples)