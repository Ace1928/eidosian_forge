import inspect
import weakref
def _get_key(self, entity):
    if inspect.ismethod(entity):
        return entity.__func__
    return entity