import functools
def _make_proxy_method(attr_name):

    def method(self, *args, **kwargs):
        return getattr(self._file, attr_name)(*args, **kwargs)
    return method