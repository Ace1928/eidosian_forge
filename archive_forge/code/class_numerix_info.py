import sys
import os
import re
import copy
import warnings
import subprocess
import textwrap
from glob import glob
from functools import reduce
from configparser import NoOptionError
from configparser import RawConfigParser as ConfigParser
from distutils.errors import DistutilsError
from distutils.dist import Distribution
import sysconfig
from numpy.distutils import log
from distutils.util import get_platform
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import (is_sequence, is_string,
from numpy.distutils.command.config import config as cmd_config
from numpy.distutils import customized_ccompiler as _customized_ccompiler
from numpy.distutils import _shell_utils
import distutils.ccompiler
import tempfile
import shutil
import platform
class numerix_info(system_info):
    section = 'numerix'

    def calc_info(self):
        which = (None, None)
        if os.getenv('NUMERIX'):
            which = (os.getenv('NUMERIX'), 'environment var')
        if which[0] is None:
            which = ('numpy', 'defaulted')
            try:
                import numpy
                which = ('numpy', 'defaulted')
            except ImportError as e:
                msg1 = str(e)
                try:
                    import Numeric
                    which = ('numeric', 'defaulted')
                except ImportError as e:
                    msg2 = str(e)
                    try:
                        import numarray
                        which = ('numarray', 'defaulted')
                    except ImportError as e:
                        msg3 = str(e)
                        log.info(msg1)
                        log.info(msg2)
                        log.info(msg3)
        which = (which[0].strip().lower(), which[1])
        if which[0] not in ['numeric', 'numarray', 'numpy']:
            raise ValueError("numerix selector must be either 'Numeric' or 'numarray' or 'numpy' but the value obtained from the %s was '%s'." % (which[1], which[0]))
        os.environ['NUMERIX'] = which[0]
        self.set_info(**get_info(which[0]))