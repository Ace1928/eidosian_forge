import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.example import ResponseExampleDocumenter
from botocore.docs.method import (
from botocore.docs.params import ResponseParamsDocumenter
from botocore.docs.sharedexample import document_shared_examples
from botocore.docs.utils import DocumentedShape, get_official_service_name
class ClientDocumenter:
    _CLIENT_METHODS_FILTERS = [_allowlist_generate_presigned_url]

    def __init__(self, client, root_docs_path, shared_examples=None):
        self._client = client
        self._client_class_name = self._client.__class__.__name__
        self._root_docs_path = root_docs_path
        self._shared_examples = shared_examples
        if self._shared_examples is None:
            self._shared_examples = {}
        self._service_name = self._client.meta.service_model.service_name

    def document_client(self, section):
        """Documents a client and its methods

        :param section: The section to write to.
        """
        self._add_title(section)
        self._add_class_signature(section)
        client_methods = self._get_client_methods()
        self._add_client_intro(section, client_methods)
        self._add_client_methods(client_methods)

    def _get_client_methods(self):
        client_methods = get_instance_public_methods(self._client)
        return self._filter_client_methods(client_methods)

    def _filter_client_methods(self, client_methods):
        filtered_methods = {}
        for method_name, method in client_methods.items():
            include = self._filter_client_method(method=method, method_name=method_name, service_name=self._service_name)
            if include:
                filtered_methods[method_name] = method
        return filtered_methods

    def _filter_client_method(self, **kwargs):
        for filter in self._CLIENT_METHODS_FILTERS:
            filter_include = filter(**kwargs)
            if filter_include is not None:
                return filter_include
        return True

    def _add_title(self, section):
        section.style.h2('Client')

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

    def _add_class_signature(self, section):
        section.style.start_sphinx_py_class(class_name=f'{self._client_class_name}.Client')

    def _add_client_creation_example(self, section):
        section.style.start_codeblock()
        section.style.new_line()
        section.write("client = session.create_client('{service}')".format(service=self._service_name))
        section.style.end_codeblock()

    def _add_client_methods(self, client_methods):
        for method_name in sorted(client_methods):
            method_doc_structure = DocumentStructure(method_name, target='html')
            self._add_client_method(method_doc_structure, method_name, client_methods[method_name])
            client_dir_path = os.path.join(self._root_docs_path, self._service_name, 'client')
            method_doc_structure.write_to_file(client_dir_path, method_name)

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

    def _is_custom_method(self, method_name):
        return method_name not in self._client.meta.method_to_api_mapping

    def _add_custom_method(self, section, method_name, method):
        document_custom_method(section, method_name, method)

    def _add_method_exceptions_list(self, section, operation_model):
        error_section = section.add_new_section('exceptions')
        error_section.style.new_line()
        error_section.style.bold('Exceptions')
        error_section.style.new_line()
        for error in operation_model.error_shapes:
            class_name = f'{self._client_class_name}.Client.exceptions.{error.name}'
            error_section.style.li(':py:class:`%s`' % class_name)

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