import os.path
import stat
from neutron_lib.tests import _base as base
from neutron_lib.utils import file
def _verify_result(self, file_mode):
    self.assertTrue(os.path.exists(self.file_name))
    with open(self.file_name) as f:
        content = f.read()
    self.assertEqual(self.data, content)
    mode = os.stat(self.file_name).st_mode
    self.assertEqual(file_mode, stat.S_IMODE(mode))