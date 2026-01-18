import inspect
from . import _api
class _ArtistKwdocLoader(dict):

    def __missing__(self, key):
        if not key.endswith(':kwdoc'):
            raise KeyError(key)
        name = key[:-len(':kwdoc')]
        from matplotlib.artist import Artist, kwdoc
        try:
            cls, = [cls for cls in _api.recursive_subclasses(Artist) if cls.__name__ == name]
        except ValueError as e:
            raise KeyError(key) from e
        return self.setdefault(key, kwdoc(cls))