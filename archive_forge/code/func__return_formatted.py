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
def _return_formatted(self, kwargs):
    self.add_path_info(kwargs)
    if 'invocation' not in kwargs:
        kwargs['invocation'] = {'module_args': self.params}
    if 'warnings' in kwargs:
        if isinstance(kwargs['warnings'], list):
            for w in kwargs['warnings']:
                self.warn(w)
        else:
            self.warn(kwargs['warnings'])
    warnings = get_warning_messages()
    if warnings:
        kwargs['warnings'] = warnings
    if 'deprecations' in kwargs:
        if isinstance(kwargs['deprecations'], list):
            for d in kwargs['deprecations']:
                if isinstance(d, SEQUENCETYPE) and len(d) == 2:
                    self.deprecate(d[0], version=d[1])
                elif isinstance(d, Mapping):
                    self.deprecate(d['msg'], version=d.get('version'), date=d.get('date'), collection_name=d.get('collection_name'))
                else:
                    self.deprecate(d)
        else:
            self.deprecate(kwargs['deprecations'])
    deprecations = get_deprecation_messages()
    if deprecations:
        kwargs['deprecations'] = deprecations
    preserved = {}
    for k, v in kwargs.items():
        if v is None or isinstance(v, bool):
            preserved[k] = v
    kwargs = remove_values(kwargs, self.no_log_values)
    kwargs.update(preserved)
    print('\n%s' % self.jsonify(kwargs))