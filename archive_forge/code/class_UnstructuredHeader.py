from types import MappingProxyType
from email import utils
from email import errors
from email import _header_value_parser as parser
class UnstructuredHeader:
    max_count = None
    value_parser = staticmethod(parser.get_unstructured)

    @classmethod
    def parse(cls, value, kwds):
        kwds['parse_tree'] = cls.value_parser(value)
        kwds['decoded'] = str(kwds['parse_tree'])