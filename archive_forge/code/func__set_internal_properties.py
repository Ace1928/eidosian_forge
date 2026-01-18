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
def _set_internal_properties(self, argument_spec=None, module_parameters=None):
    if argument_spec is None:
        argument_spec = self.argument_spec
    if module_parameters is None:
        module_parameters = self.params
    for k in PASS_VARS:
        param_key = '_ansible_%s' % k
        if param_key in module_parameters:
            if k in PASS_BOOLS:
                setattr(self, PASS_VARS[k][0], self.boolean(module_parameters[param_key]))
            else:
                setattr(self, PASS_VARS[k][0], module_parameters[param_key])
            if param_key in self.params:
                del self.params[param_key]
        elif not hasattr(self, PASS_VARS[k][0]):
            setattr(self, PASS_VARS[k][0], PASS_VARS[k][1])