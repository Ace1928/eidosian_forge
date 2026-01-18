import binascii
import math
import socket
import struct
from typing import Any, Dict, Optional, Union
import dns.enum
import dns.inet
import dns.rdata
import dns.wire
class EDEOption(Option):
    """Extended DNS Error (EDE, RFC8914)"""
    _preserve_case = {'DNSKEY', 'DS', 'DNSSEC', 'RRSIGs', 'NSEC', 'NXDOMAIN'}

    def __init__(self, code: Union[EDECode, str], text: Optional[str]=None):
        """*code*, a ``dns.edns.EDECode`` or ``str``, the info code of the
        extended error.

        *text*, a ``str`` or ``None``, specifying additional information about
        the error.
        """
        super().__init__(OptionType.EDE)
        self.code = EDECode.make(code)
        if text is not None and (not isinstance(text, str)):
            raise ValueError('text must be string or None')
        self.text = text

    def to_text(self) -> str:
        output = f'EDE {self.code}'
        if self.code in EDECode:
            desc = EDECode.to_text(self.code)
            desc = ' '.join((word if word in self._preserve_case else word.title() for word in desc.split('_')))
            output += f' ({desc})'
        if self.text is not None:
            output += f': {self.text}'
        return output

    def to_wire(self, file: Optional[Any]=None) -> Optional[bytes]:
        value = struct.pack('!H', self.code)
        if self.text is not None:
            value += self.text.encode('utf8')
        if file:
            file.write(value)
            return None
        else:
            return value

    @classmethod
    def from_wire_parser(cls, otype: Union[OptionType, str], parser: 'dns.wire.Parser') -> Option:
        code = EDECode.make(parser.get_uint16())
        text = parser.get_remaining()
        if text:
            if text[-1] == 0:
                text = text[:-1]
            btext = text.decode('utf8')
        else:
            btext = None
        return cls(code, btext)