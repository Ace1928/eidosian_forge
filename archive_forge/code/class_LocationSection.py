import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
class LocationSection(Section):

    def __init__(self, section, extra_path, branch_name=None):
        super().__init__(section.id, section.options)
        self.extra_path = extra_path
        if branch_name is None:
            branch_name = ''
        self.locals = {'relpath': extra_path, 'basename': urlutils.basename(extra_path), 'branchname': branch_name}

    def get(self, name, default=None, expand=True):
        value = super().get(name, default)
        if value is not None and expand:
            policy_name = self.get(name + ':policy', None)
            policy = _policy_value.get(policy_name, POLICY_NONE)
            if policy == POLICY_APPENDPATH:
                value = urlutils.join(value, self.extra_path)
            chunks = []
            for is_ref, chunk in iter_option_refs(value):
                if not is_ref:
                    chunks.append(chunk)
                else:
                    ref = chunk[1:-1]
                    if ref in self.locals:
                        chunks.append(self.locals[ref])
                    else:
                        chunks.append(chunk)
            value = ''.join(chunks)
        return value