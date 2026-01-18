from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import itertools
import logging
import os
import sys
from xml.dom import minidom
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _helpers
from absl.flags import _validators_classes
import six
def _render_flag_list(self, flaglist, output_lines, prefix='  '):
    fl = self._flags()
    special_fl = _helpers.SPECIAL_FLAGS._flags()
    flaglist = [(flag.name, flag) for flag in flaglist]
    flaglist.sort()
    flagset = {}
    for name, flag in flaglist:
        if fl.get(name, None) != flag and special_fl.get(name, None) != flag:
            continue
        if flag in flagset:
            continue
        flagset[flag] = 1
        flaghelp = ''
        if flag.short_name:
            flaghelp += '-%s,' % flag.short_name
        if flag.boolean:
            flaghelp += '--[no]%s:' % flag.name
        else:
            flaghelp += '--%s:' % flag.name
        flaghelp += ' '
        if flag.help:
            flaghelp += flag.help
        flaghelp = _helpers.text_wrap(flaghelp, indent=prefix + '  ', firstline_indent=prefix)
        if flag.default_as_str:
            flaghelp += '\n'
            flaghelp += _helpers.text_wrap('(default: %s)' % flag.default_as_str, indent=prefix + '  ')
        if flag.parser.syntactic_help:
            flaghelp += '\n'
            flaghelp += _helpers.text_wrap('(%s)' % flag.parser.syntactic_help, indent=prefix + '  ')
        output_lines.append(flaghelp)