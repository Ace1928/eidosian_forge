import collections
import gast
from tensorflow.python.autograph.pyct import gast_util
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
def _ensure_fields_in_anf(self, node, parent=None, super_field=None):
    for field in node._fields:
        if field.startswith('__'):
            continue
        parent_supplied = node if parent is None else parent
        field_supplied = field if super_field is None else super_field
        setattr(node, field, self._ensure_node_in_anf(parent_supplied, field_supplied, getattr(node, field)))
    return node