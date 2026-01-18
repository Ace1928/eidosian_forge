from paste.registry import RegistryManager, StackedObjectProxy
def _current_obj(self):
    try:
        return super(DispatchingConfig, self)._current_obj()
    except TypeError:
        if self._process_configs:
            return self._process_configs[-1]
        raise AttributeError('No configuration has been registered for this process or thread')