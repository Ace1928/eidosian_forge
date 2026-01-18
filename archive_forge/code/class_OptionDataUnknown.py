import struct
from os_ken.lib import stringify
from os_ken.lib import type_desc
from . import packet_base
from . import ether_types
@Option.register_unknown_type()
class OptionDataUnknown(Option):
    """
    Unknown Option Class and Type specific Option
    """

    def __init__(self, buf, option_class=None, type_=None, length=0):
        super(OptionDataUnknown, self).__init__(option_class=option_class, type_=type_, length=length)
        self.buf = buf

    @classmethod
    def parse_value(cls, buf):
        return {'buf': buf}

    def serialize_value(self):
        return self.buf