from types import MappingProxyType
from email import utils
from email import errors
from email import _header_value_parser as parser
class MessageIDHeader:
    max_count = 1
    value_parser = staticmethod(parser.parse_message_id)

    @classmethod
    def parse(cls, value, kwds):
        kwds['parse_tree'] = parse_tree = cls.value_parser(value)
        kwds['decoded'] = str(parse_tree)
        kwds['defects'].extend(parse_tree.all_defects)