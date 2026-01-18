from botocore.docs.shape import ShapeDocumenter
from botocore.docs.utils import py_type_name
def _append_idempotency_documentation(self, section):
    docstring = 'This field is autopopulated if not provided.'
    section.write(docstring)