from io import BytesIO
import sys
import pytest
from srsly.msgpack import Unpacker, packb, OutOfData, ExtType
class MyUnpacker(Unpacker):

    def __init__(self):
        super(MyUnpacker, self).__init__(ext_hook=self._hook, raw=False)

    def _hook(self, code, data):
        if code == 1:
            return int(data)
        else:
            return ExtType(code, data)