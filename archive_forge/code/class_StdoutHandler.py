import inspect
import io
import logging
import re
import sys
import textwrap
from pyomo.version.info import releaselevel
from pyomo.common.deprecation import deprecated
from pyomo.common.fileutils import PYOMO_ROOT_DIR
from pyomo.common.formatting import wrap_reStructuredText
class StdoutHandler(logging.StreamHandler):
    """A logging handler that emits to the current value of sys.stdout"""

    def flush(self):
        self.stream = sys.stdout
        super(StdoutHandler, self).flush()

    def emit(self, record):
        self.stream = sys.stdout
        super(StdoutHandler, self).emit(record)