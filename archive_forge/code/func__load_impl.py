from mako import util
def _load_impl(self, name):
    return _cache_plugins.load(name)(self)