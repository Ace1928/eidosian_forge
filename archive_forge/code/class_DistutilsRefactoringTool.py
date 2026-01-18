import os
import re
import importlib.util
import string
import sys
import distutils
from distutils.errors import DistutilsPlatformError
from distutils.dep_util import newer
from distutils.spawn import spawn
from distutils import log
from distutils.errors import DistutilsByteCompileError
from distutils.util import byte_compile
class DistutilsRefactoringTool(RefactoringTool):

    def log_error(self, msg, *args, **kw):
        log.error(msg, *args)

    def log_message(self, msg, *args):
        log.info(msg, *args)

    def log_debug(self, msg, *args):
        log.debug(msg, *args)