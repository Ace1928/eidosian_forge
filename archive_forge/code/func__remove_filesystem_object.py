import os
import time
import tempfile
import logging
import shutil
import weakref
from pyomo.common.dependencies import attempt_import, pyutilib_available
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import TempfileContextError
from pyomo.common.multithread import MultiThreadWrapperWithMain
def _remove_filesystem_object(self, name):
    if not os.path.exists(name):
        return
    if os.path.isfile(name) or os.path.islink(name):
        try:
            os.remove(name)
        except WindowsError:
            try:
                time.sleep(1)
                os.remove(name)
            except WindowsError:
                if deletion_errors_are_fatal:
                    raise
                else:
                    logger = logging.getLogger(__name__)
                    logger.warning('Unable to delete temporary file %s' % (name,))
        return
    assert os.path.isdir(name)
    shutil.rmtree(name, ignore_errors=not deletion_errors_are_fatal)