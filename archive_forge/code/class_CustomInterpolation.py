import copy
import inspect
import io
import re
import warnings
from configparser import (
from dataclasses import dataclass
from pathlib import Path
from types import GeneratorType
from typing import (
import srsly
from .util import SimpleFrozenDict, SimpleFrozenList  # noqa: F401
class CustomInterpolation(ExtendedInterpolation):

    def before_read(self, parser, section, option, value):
        try:
            json_value = srsly.json_loads(value)
            if isinstance(json_value, str) and json_value not in JSON_EXCEPTIONS:
                value = json_value
        except ValueError:
            if value and value[0] == value[-1] == "'":
                warnings.warn(f'The value [{value}] seems to be single-quoted, but values use JSON formatting, which requires double quotes.')
        except Exception:
            pass
        return super().before_read(parser, section, option, value)

    def before_get(self, parser, section, option, value, defaults):
        L = []
        self.interpolate(parser, option, L, value, section, defaults, 1)
        return ''.join(L)

    def interpolate(self, parser, option, accum, rest, section, map, depth):
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
                    err = f'bad interpolation variable reference {rest}'
                    raise InterpolationSyntaxError(option, section, err)
                orig_var = m.group(1)
                path = orig_var.replace(':', '.').rsplit('.', 1)
                rest = rest[m.end():]
                sect = section
                opt = option
                try:
                    if len(path) == 1:
                        opt = parser.optionxform(path[0])
                        if opt in map:
                            v = map[opt]
                        else:
                            section_name = parser[parser.optionxform(path[0])]._name
                            v = self._get_section_name(section_name)
                    elif len(path) == 2:
                        sect = path[0]
                        opt = parser.optionxform(path[1])
                        fallback = '__FALLBACK__'
                        v = parser.get(sect, opt, raw=True, fallback=fallback)
                        if v == fallback:
                            v = self._get_section_name(parser[f'{sect}.{opt}']._name)
                    else:
                        err = f"More than one ':' found: {rest}"
                        raise InterpolationSyntaxError(option, section, err)
                except (KeyError, NoSectionError, NoOptionError):
                    raise InterpolationMissingOptionError(option, section, rawval, orig_var) from None
                if '$' in v:
                    new_map = dict(parser.items(sect, raw=True))
                    self.interpolate(parser, opt, accum, v, sect, new_map, depth + 1)
                else:
                    accum.append(v)
            else:
                err = "'$' must be followed by '$' or '{', found: %r" % (rest,)
                raise InterpolationSyntaxError(option, section, err)

    def _get_section_name(self, name: str) -> str:
        """Generate the name of a section. Note that we use a quoted string here
        so we can use section references within lists and load the list as
        JSON. Since section references can't be used within strings, we don't
        need the quoted vs. unquoted distinction like we do for variables.

        Examples (assuming section = {"foo": 1}):
            - value: ${section.foo} -> value: 1
            - value: "hello ${section.foo}" -> value: "hello 1"
            - value: ${section} -> value: {"foo": 1}
            - value: "${section}" -> value: {"foo": 1}
            - value: "hello ${section}" -> invalid
        """
        return f'"{SECTION_PREFIX}{name}"'