from botocore.docs.shape import ShapeDocumenter
from botocore.docs.utils import py_type_name
def _add_special_trait_documentation(self, section, shape):
    if 'idempotencyToken' in shape.metadata:
        self._append_idempotency_documentation(section)