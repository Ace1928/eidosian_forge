from __future__ import absolute_import, division, print_function
import sys
import __main__
import atexit
import errno
import datetime
import grp
import fcntl
import locale
import os
import pwd
import platform
import re
import select
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import traceback
import types
from itertools import chain, repeat
from ansible.module_utils.compat import selectors
from ._text import to_native, to_bytes, to_text
from ansible.module_utils.common.text.converters import (
from ansible.module_utils.common.arg_spec import ModuleArgumentSpecValidator
from ansible.module_utils.common.text.formatters import (
import hashlib
from ansible.module_utils.six.moves.collections_abc import (
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.file import (
from ansible.module_utils.common.sys_info import (
from ansible.module_utils.pycompat24 import get_exception, literal_eval
from ansible.module_utils.common.parameters import (
from ansible.module_utils.errors import AnsibleFallbackNotFound, AnsibleValidationErrorMultiple, UnsupportedError
from ansible.module_utils.six import (
from ansible.module_utils.six.moves import map, reduce, shlex_quote
from ansible.module_utils.common.validation import (
from ansible.module_utils.common._utils import get_all_subclasses as _get_all_subclasses
from ansible.module_utils.parsing.convert_bool import BOOLEANS, BOOLEANS_FALSE, BOOLEANS_TRUE, boolean
from ansible.module_utils.common.warnings import (
def _log_invocation(self):
    """ log that ansible ran the module """
    log_args = dict()
    for param in self.params:
        canon = self.aliases.get(param, param)
        arg_opts = self.argument_spec.get(canon, {})
        no_log = arg_opts.get('no_log', None)
        if no_log is None and PASSWORD_MATCH.search(param):
            log_args[param] = 'NOT_LOGGING_PASSWORD'
            self.warn('Module did not set no_log for %s' % param)
        elif self.boolean(no_log):
            log_args[param] = 'NOT_LOGGING_PARAMETER'
        else:
            param_val = self.params[param]
            if not isinstance(param_val, (text_type, binary_type)):
                param_val = str(param_val)
            elif isinstance(param_val, text_type):
                param_val = param_val.encode('utf-8')
            log_args[param] = heuristic_log_sanitize(param_val, self.no_log_values)
    msg = ['%s=%s' % (to_native(arg), to_native(val)) for arg, val in log_args.items()]
    if msg:
        msg = 'Invoked with %s' % ' '.join(msg)
    else:
        msg = 'Invoked'
    self.log(msg, log_args=log_args)