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
class _GlobalLogFilter(object):

    def __init__(self):
        self.logger = logging.getLogger()

    def filter(self, record):
        return not self.logger.handlers