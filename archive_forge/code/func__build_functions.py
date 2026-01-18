from functools import wraps
def _build_functions(self):
    for name, callback in self._cached_base_callbacks.items():
        for plugin in reversed(self._registered_plugins):
            try:
                func = getattr(plugin, name)
            except AttributeError:
                pass
            else:
                callback = func(callback)
        self._built_functions[name] = callback