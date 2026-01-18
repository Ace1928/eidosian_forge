from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import functools
import io
import itertools
import os
import re
import sys
import warnings
def _interpolate_some(self, parser, option, accum, rest, section, map, depth):
    rawval = parser.get(section, option, raw=True, fallback=rest)
    if depth > MAX_INTERPOLATION_DEPTH:
        raise InterpolationDepthError(option, section, rawval)
    while rest:
        p = rest.find('$')
        if p < 0:
            accum.append(rest)
            return
        if p > 0:
            accum.append(rest[:p])
            rest = rest[p:]
        c = rest[1:2]
        if c == '$':
            accum.append('$')
            rest = rest[2:]
        elif c == '{':
            m = self._KEYCRE.match(rest)
            if m is None:
                raise InterpolationSyntaxError(option, section, 'bad interpolation variable reference %r' % rest)
            path = m.group(1).split(':')
            rest = rest[m.end():]
            sect = section
            opt = option
            try:
                if len(path) == 1:
                    opt = parser.optionxform(path[0])
                    v = map[opt]
                elif len(path) == 2:
                    sect = path[0]
                    opt = parser.optionxform(path[1])
                    v = parser.get(sect, opt, raw=True)
                else:
                    raise InterpolationSyntaxError(option, section, "More than one ':' found: %r" % (rest,))
            except (KeyError, NoSectionError, NoOptionError):
                raise InterpolationMissingOptionError(option, section, rawval, ':'.join(path)) from None
            if '$' in v:
                self._interpolate_some(parser, opt, accum, v, sect, dict(parser.items(sect, raw=True)), depth + 1)
            else:
                accum.append(v)
        else:
            raise InterpolationSyntaxError(option, section, "'$' must be followed by '$' or '{', found: %r" % (rest,))