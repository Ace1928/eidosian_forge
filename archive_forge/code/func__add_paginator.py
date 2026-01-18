import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.method import document_model_driven_method
from botocore.docs.utils import DocumentedShape
from botocore.utils import get_service_module_name
def _add_paginator(self, section, paginator_name):
    breadcrumb_section = section.add_new_section('breadcrumb')
    breadcrumb_section.style.ref(self._client_class_name, f'../../{self._service_name}')
    breadcrumb_section.write(f' / Paginator / {paginator_name}')
    section.add_title_section(paginator_name)
    paginator_section = section.add_new_section(paginator_name)
    paginator_section.style.start_sphinx_py_class(class_name=f'{self._client_class_name}.Paginator.{paginator_name}')
    paginator_section.style.start_codeblock()
    paginator_section.style.new_line()
    paginator_section.write(f"paginator = client.get_paginator('{xform_name(paginator_name)}')")
    paginator_section.style.end_codeblock()
    paginator_section.style.new_line()
    paginator_config = self._service_paginator_model.get_paginator(paginator_name)
    document_paginate_method(section=paginator_section, paginator_name=paginator_name, event_emitter=self._client.meta.events, service_model=self._client.meta.service_model, paginator_config=paginator_config)