import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.method import document_model_driven_method
from botocore.docs.utils import DocumentedShape
from botocore.utils import get_service_module_name
class WaiterDocumenter:

    def __init__(self, client, service_waiter_model, root_docs_path):
        self._client = client
        self._client_class_name = self._client.__class__.__name__
        self._service_name = self._client.meta.service_model.service_name
        self._service_waiter_model = service_waiter_model
        self._root_docs_path = root_docs_path
        self._USER_GUIDE_LINK = 'https://boto3.amazonaws.com/v1/documentation/api/latest/guide/clients.html#waiters'

    def document_waiters(self, section):
        """Documents the various waiters for a service.

        :param section: The section to write to.
        """
        section.style.h2('Waiters')
        self._add_overview(section)
        section.style.new_line()
        section.writeln('The available waiters are:')
        section.style.toctree()
        for waiter_name in self._service_waiter_model.waiter_names:
            section.style.tocitem(f'{self._service_name}/waiter/{waiter_name}')
            waiter_doc_structure = DocumentStructure(waiter_name, target='html')
            self._add_single_waiter(waiter_doc_structure, waiter_name)
            waiter_dir_path = os.path.join(self._root_docs_path, self._service_name, 'waiter')
            waiter_doc_structure.write_to_file(waiter_dir_path, waiter_name)

    def _add_single_waiter(self, section, waiter_name):
        breadcrumb_section = section.add_new_section('breadcrumb')
        breadcrumb_section.style.ref(self._client_class_name, f'../../{self._service_name}')
        breadcrumb_section.write(f' / Waiter / {waiter_name}')
        section.add_title_section(waiter_name)
        waiter_section = section.add_new_section(waiter_name)
        waiter_section.style.start_sphinx_py_class(class_name=f'{self._client_class_name}.Waiter.{waiter_name}')
        waiter_section.style.start_codeblock()
        waiter_section.style.new_line()
        waiter_section.write("waiter = client.get_waiter('%s')" % xform_name(waiter_name))
        waiter_section.style.end_codeblock()
        waiter_section.style.new_line()
        document_wait_method(section=waiter_section, waiter_name=waiter_name, event_emitter=self._client.meta.events, service_model=self._client.meta.service_model, service_waiter_model=self._service_waiter_model)

    def _add_overview(self, section):
        section.style.new_line()
        section.write('Waiters are available on a client instance via the ``get_waiter`` method. For more detailed instructions and examples on the usage or waiters, see the waiters ')
        section.style.external_link(title='user guide', link=self._USER_GUIDE_LINK)
        section.write('.')
        section.style.new_line()