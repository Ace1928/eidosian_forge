import array
from typing import (
def _parse_uncompressed(self, bits: Optional[str]) -> BitParserState:
    if not bits:
        raise self.InvalidData
    if bits.startswith('T'):
        self._accept = self._parse_mode
        self._color = int(bits[1])
        self._do_uncompressed(bits[2:])
        return self.MODE
    else:
        self._do_uncompressed(bits)
        return self.UNCOMPRESSED