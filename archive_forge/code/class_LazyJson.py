from ._base import *
class LazyJson:
    serializer = json
    parser = json.Parser()

    @classmethod
    def dumps(cls, obj, *args, **kwargs):
        return LazyJson.serializer.dumps(obj, *args, **kwargs)

    @classmethod
    def dump(cls, obj, fileio, indent=2, *args, **kwargs):
        return LazyJson.serializer.dump(obj, *args, fp=fileio, indent=indent, **kwargs)

    @classmethod
    def loads(cls, jsonstring, use_parser=True, recursive=False, ignore_errors=True, *args, **kwargs):
        if use_parser:
            try:
                return LazyJson.parser.parse(jsonstring, recursive=recursive)
            except Exception as e:
                if not ignore_errors:
                    raise ValueError(str(e))
        try:
            return LazyJson.serializer.loads(jsonstring, *args, **kwargs)
        except Exception as e:
            if not ignore_errors:
                raise ValueError(str(e))
        return None

    @classmethod
    def load(cls, fileio, use_parser=True, recursive=False, ignore_errors=True, *args, **kwargs):
        if use_parser:
            try:
                return LazyJson.parser.parse(fileio, recursive=recursive)
            except Exception as e:
                if not ignore_errors:
                    raise ValueError(str(e))
        try:
            return LazyJson.serializer.load(fileio, *args, **kwargs)
        except Exception as e:
            if not ignore_errors:
                raise ValueError(str(e))
        return None