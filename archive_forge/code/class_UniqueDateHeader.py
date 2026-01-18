from types import MappingProxyType
from email import utils
from email import errors
from email import _header_value_parser as parser
class UniqueDateHeader(DateHeader):
    max_count = 1