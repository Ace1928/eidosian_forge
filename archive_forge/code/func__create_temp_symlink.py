import os
import tempfile
from os_win import constants
from os_win.tests.functional import test_base
from os_win import utilsfactory
def _create_temp_symlink(self, target, target_is_dir):
    f = tempfile.TemporaryFile(prefix='oswin_vhdtest_link_')
    f.close()
    self._pathutils.create_sym_link(f.name, target, target_is_dir)
    if target_is_dir:
        self.addCleanup(os.rmdir, f.name)
    else:
        self.addCleanup(os.unlink, f.name)
    return f.name