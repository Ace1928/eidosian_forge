from types import MappingProxyType
from email import utils
from email import errors
from email import _header_value_parser as parser
class ParameterizedMIMEHeader:
    max_count = 1

    @classmethod
    def parse(cls, value, kwds):
        kwds['parse_tree'] = parse_tree = cls.value_parser(value)
        kwds['decoded'] = str(parse_tree)
        kwds['defects'].extend(parse_tree.all_defects)
        if parse_tree.params is None:
            kwds['params'] = {}
        else:
            kwds['params'] = {utils._sanitize(name).lower(): utils._sanitize(value) for name, value in parse_tree.params}

    def init(self, *args, **kw):
        self._params = kw.pop('params')
        super().init(*args, **kw)

    @property
    def params(self):
        return MappingProxyType(self._params)