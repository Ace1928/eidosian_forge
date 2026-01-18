from ._base import *
class LazyPickler:
    serializer = pickler
    protocol = pickler.HIGHEST_PROTOCOL

    @classmethod
    def dumps(cls, obj, *args, **kwargs):
        return LazyPickler.serializer.dumps(obj, *args, protocol=LazyPickler.protocol, **kwargs)

    @classmethod
    def loads(cls, data, *args, **kwargs):
        return LazyPickler.serializer.loads(data, *args, **kwargs)

    @classmethod
    def dump(cls, obj, fileio, *args, **kwargs):
        data = LazyPickler.dumps(obj, *args, **kwargs)
        fileio.write(data)
        fileio.flush()

    @classmethod
    def load(cls, fileio, *args, **kwargs):
        return LazyPickler.loads(fileio.read(), *args, **kwargs)