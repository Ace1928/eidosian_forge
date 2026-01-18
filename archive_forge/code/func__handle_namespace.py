from _pydev_bundle.pydev_imports import execfile
from _pydevd_bundle import pydevd_dont_trace
import types
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_constants import get_global_debugger
def _handle_namespace(self, namespace, is_class_namespace=False):
    on_finish = None
    if is_class_namespace:
        xreload_after_update = getattr(namespace, '__xreload_after_reload_update__', None)
        if xreload_after_update is not None:
            self.found_change = True
            on_finish = lambda: xreload_after_update()
    elif '__xreload_after_reload_update__' in namespace:
        xreload_after_update = namespace['__xreload_after_reload_update__']
        self.found_change = True
        on_finish = lambda: xreload_after_update(namespace)
    if on_finish is not None:
        self._on_finish_callbacks.append(on_finish)