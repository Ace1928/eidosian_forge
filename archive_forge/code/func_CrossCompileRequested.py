import errno
import filecmp
import os.path
import re
import tempfile
import sys
import subprocess
from collections.abc import MutableSet
def CrossCompileRequested():
    return os.environ.get('GYP_CROSSCOMPILE') or os.environ.get('AR_host') or os.environ.get('CC_host') or os.environ.get('CXX_host') or os.environ.get('AR_target') or os.environ.get('CC_target') or os.environ.get('CXX_target')