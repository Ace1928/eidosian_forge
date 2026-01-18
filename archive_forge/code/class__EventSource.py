from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import add_metaclass
class _EventSource:

    def __init__(self):
        self._handlers = set()

    def __iadd__(self, handler):
        if not callable(handler):
            raise ValueError('handler must be callable')
        self._handlers.add(handler)
        return self

    def __isub__(self, handler):
        try:
            self._handlers.remove(handler)
        except KeyError:
            pass
        return self

    def _on_exception(self, handler, exc, *args, **kwargs):
        return True

    def fire(self, *args, **kwargs):
        for h in self._handlers:
            try:
                h(*args, **kwargs)
            except Exception as ex:
                if self._on_exception(h, ex, *args, **kwargs):
                    raise