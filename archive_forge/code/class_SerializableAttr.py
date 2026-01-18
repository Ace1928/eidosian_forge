import pytest
import srsly
from thinc.api import (
class SerializableAttr:
    value = 'foo'

    def to_bytes(self):
        return self.value.encode('utf8')

    def from_bytes(self, data):
        self.value = f'{data.decode('utf8')} from bytes'
        return self