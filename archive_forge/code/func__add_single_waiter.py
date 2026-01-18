import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.method import document_model_driven_method
from botocore.docs.utils import DocumentedShape
from botocore.utils import get_service_module_name
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