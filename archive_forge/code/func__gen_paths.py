import sys
import os
import re
import pathlib
import contextlib
import logging
from email import message_from_file
from .errors import (
from .fancy_getopt import FancyGetopt, translate_longopt
from .util import check_environ, strtobool, rfc822_escape
from ._log import log
from .debug import DEBUG
def _gen_paths(self):
    sys_dir = pathlib.Path(sys.modules['distutils'].__file__).parent
    yield (sys_dir / 'distutils.cfg')
    prefix = '.' * (os.name == 'posix')
    filename = prefix + 'pydistutils.cfg'
    if self.want_user_cfg:
        yield (pathlib.Path('~').expanduser() / filename)
    yield pathlib.Path('setup.cfg')
    with contextlib.suppress(TypeError):
        yield pathlib.Path(os.getenv('DIST_EXTRA_CONFIG'))