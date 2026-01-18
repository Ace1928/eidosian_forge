from parso.python import tree
from jedi import debug
from jedi.inference.helpers import is_string
def add_attribute_error(name_context, lookup_value, name):
    message = 'AttributeError: %s has no attribute %s.' % (lookup_value, name)
    typ = Error
    if lookup_value.is_instance() and (not lookup_value.is_compiled()):
        if _check_for_setattr(lookup_value):
            typ = Warning
    payload = (lookup_value, name)
    add(name_context, 'attribute-error', name, message, typ, payload)