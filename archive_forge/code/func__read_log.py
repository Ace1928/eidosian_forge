from io import BytesIO
from dulwich.tests import TestCase
from ..objects import ZERO_SHA
from ..reflog import (
def _read_log(self):
    self.f.seek(0)
    return list(read_reflog(self.f))