from __future__ import (absolute_import, division, print_function)
from ansible.utils.sentinel import Sentinel
class FieldAttribute(Attribute):

    def __init__(self, extend=False, prepend=False, **kwargs):
        super().__init__(**kwargs)
        self.extend = extend
        self.prepend = prepend

    def __get__(self, obj, obj_type=None):
        if getattr(obj, '_squashed', False) or getattr(obj, '_finalized', False):
            value = getattr(obj, f'_{self.name}', Sentinel)
        else:
            try:
                value = obj._get_parent_attribute(self.name)
            except AttributeError:
                method = f'_get_attr_{self.name}'
                if hasattr(obj, method):
                    if getattr(obj, '_squashed', False):
                        value = getattr(obj, f'_{self.name}', Sentinel)
                    else:
                        value = getattr(obj, method)()
                else:
                    value = getattr(obj, f'_{self.name}', Sentinel)
        if value is Sentinel:
            value = self.default
            if callable(value):
                value = value()
        return value