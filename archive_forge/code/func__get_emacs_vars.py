import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _get_emacs_vars(self, text):
    """Return a dictionary of emacs-style local variables.

        Parsing is done loosely according to this spec (and according to
        some in-practice deviations from this):
        http://www.gnu.org/software/emacs/manual/html_node/emacs/Specifying-File-Variables.html#Specifying-File-Variables
        """
    emacs_vars = {}
    SIZE = pow(2, 13)
    head = text[:SIZE]
    if '-*-' in head:
        match = self._emacs_oneliner_vars_pat.search(head)
        if match:
            emacs_vars_str = match.group(2)
            assert '\n' not in emacs_vars_str
            emacs_var_strs = [s.strip() for s in emacs_vars_str.split(';') if s.strip()]
            if len(emacs_var_strs) == 1 and ':' not in emacs_var_strs[0]:
                emacs_vars['mode'] = emacs_var_strs[0].strip()
            else:
                for emacs_var_str in emacs_var_strs:
                    try:
                        variable, value = emacs_var_str.strip().split(':', 1)
                    except ValueError:
                        log.debug('emacs variables error: malformed -*- line: %r', emacs_var_str)
                        continue
                    emacs_vars[variable.lower()] = value.strip()
    tail = text[-SIZE:]
    if 'Local Variables' in tail:
        match = self._emacs_local_vars_pat.search(tail)
        if match:
            prefix = match.group('prefix')
            suffix = match.group('suffix')
            lines = match.group('content').splitlines(0)
            for i, line in enumerate(lines):
                if not line.startswith(prefix):
                    log.debug("emacs variables error: line '%s' does not use proper prefix '%s'" % (line, prefix))
                    return {}
                if i != len(lines) - 1 and (not line.endswith(suffix)):
                    log.debug("emacs variables error: line '%s' does not use proper suffix '%s'" % (line, suffix))
                    return {}
            continued_for = None
            for line in lines[:-1]:
                if prefix:
                    line = line[len(prefix):]
                if suffix:
                    line = line[:-len(suffix)]
                line = line.strip()
                if continued_for:
                    variable = continued_for
                    if line.endswith('\\'):
                        line = line[:-1].rstrip()
                    else:
                        continued_for = None
                    emacs_vars[variable] += ' ' + line
                else:
                    try:
                        variable, value = line.split(':', 1)
                    except ValueError:
                        log.debug("local variables error: missing colon in local variables entry: '%s'" % line)
                        continue
                    value = value.strip()
                    if value.endswith('\\'):
                        value = value[:-1].rstrip()
                        continued_for = variable
                    else:
                        continued_for = None
                    emacs_vars[variable] = value
    for var, val in list(emacs_vars.items()):
        if len(val) > 1 and (val.startswith('"') and val.endswith('"') or (val.startswith('"') and val.endswith('"'))):
            emacs_vars[var] = val[1:-1]
    return emacs_vars