from __future__ import absolute_import
import cython
import hashlib
import operator
import os
import re
import shutil
import textwrap
from string import Template
from functools import partial
from contextlib import closing, contextmanager
from collections import defaultdict
from . import Naming
from . import Options
from . import DebugFlags
from . import StringEncoding
from .. import Utils
from .Scanning import SourceDescriptor
from ..StringIOTree import StringIOTree
class UtilityCodeBase(object):
    """
    Support for loading utility code from a file.

    Code sections in the file can be specified as follows:

        ##### MyUtility.proto #####

        [proto declarations]

        ##### MyUtility.init #####

        [code run at module initialization]

        ##### MyUtility #####
        #@requires: MyOtherUtility
        #@substitute: naming

        [definitions]

        ##### MyUtility #####
        #@substitute: tempita

        [requires tempita substitution
         - context can't be specified here though so only
           tempita utility that requires no external context
           will benefit from this tag
         - only necessary when @required from non-tempita code]

    for prototypes and implementation respectively.  For non-python or
    -cython files backslashes should be used instead.  5 to 30 comment
    characters may be used on either side.

    If the @cname decorator is not used and this is a CythonUtilityCode,
    one should pass in the 'name' keyword argument to be used for name
    mangling of such entries.
    """
    is_cython_utility = False
    _utility_cache = {}

    @classmethod
    def _add_utility(cls, utility, type, lines, begin_lineno, tags=None):
        if utility is None:
            return
        code = '\n'.join(lines)
        if tags and 'substitute' in tags and ('naming' in tags['substitute']):
            try:
                code = Template(code).substitute(vars(Naming))
            except (KeyError, ValueError) as e:
                raise RuntimeError("Error parsing templated utility code of type '%s' at line %d: %s" % (type, begin_lineno, e))
        code = '\n' * begin_lineno + code
        if type == 'proto':
            utility[0] = code
        elif type == 'impl':
            utility[1] = code
        else:
            all_tags = utility[2]
            all_tags[type] = code
        if tags:
            all_tags = utility[2]
            for name, values in tags.items():
                all_tags.setdefault(name, set()).update(values)

    @classmethod
    def load_utilities_from_file(cls, path):
        utilities = cls._utility_cache.get(path)
        if utilities:
            return utilities
        _, ext = os.path.splitext(path)
        if ext in ('.pyx', '.py', '.pxd', '.pxi'):
            comment = '#'
            strip_comments = partial(re.compile('^\\s*#(?!\\s*cython\\s*:).*').sub, '')
            rstrip = StringEncoding._unicode.rstrip
        else:
            comment = '/'
            strip_comments = partial(re.compile('^\\s*//.*|/\\*[^*]*\\*/').sub, '')
            rstrip = partial(re.compile('\\s+(\\\\?)$').sub, '\\1')
        match_special = re.compile('^%(C)s{5,30}\\s*(?P<name>(?:\\w|\\.)+)\\s*%(C)s{5,30}|^%(C)s+@(?P<tag>\\w+)\\s*:\\s*(?P<value>(?:\\w|[.:])+)' % {'C': comment}).match
        match_type = re.compile('(.+)[.](proto(?:[.]\\S+)?|impl|init|cleanup)$').match
        all_lines = read_utilities_hook(path)
        utilities = defaultdict(lambda: [None, None, {}])
        lines = []
        tags = defaultdict(set)
        utility = type = None
        begin_lineno = 0
        for lineno, line in enumerate(all_lines):
            m = match_special(line)
            if m:
                if m.group('name'):
                    cls._add_utility(utility, type, lines, begin_lineno, tags)
                    begin_lineno = lineno + 1
                    del lines[:]
                    tags.clear()
                    name = m.group('name')
                    mtype = match_type(name)
                    if mtype:
                        name, type = mtype.groups()
                    else:
                        type = 'impl'
                    utility = utilities[name]
                else:
                    tags[m.group('tag')].add(m.group('value'))
                    lines.append('')
            else:
                lines.append(rstrip(strip_comments(line)))
        if utility is None:
            raise ValueError('Empty utility code file')
        cls._add_utility(utility, type, lines, begin_lineno, tags)
        utilities = dict(utilities)
        cls._utility_cache[path] = utilities
        return utilities

    @classmethod
    def load(cls, util_code_name, from_file, **kwargs):
        """
        Load utility code from a file specified by from_file (relative to
        Cython/Utility) and name util_code_name.
        """
        if '::' in util_code_name:
            from_file, util_code_name = util_code_name.rsplit('::', 1)
        assert from_file
        utilities = cls.load_utilities_from_file(from_file)
        proto, impl, tags = utilities[util_code_name]
        if tags:
            if 'substitute' in tags and 'tempita' in tags['substitute']:
                if not issubclass(cls, TempitaUtilityCode):
                    return TempitaUtilityCode.load(util_code_name, from_file, **kwargs)
            orig_kwargs = kwargs.copy()
            for name, values in tags.items():
                if name in kwargs:
                    continue
                if name == 'requires':
                    if orig_kwargs:
                        values = [cls.load(dep, from_file, **orig_kwargs) for dep in sorted(values)]
                    else:
                        values = [cls.load_cached(dep, from_file) for dep in sorted(values)]
                elif name == 'substitute':
                    values = values - {'naming', 'tempita'}
                    if not values:
                        continue
                elif not values:
                    values = None
                elif len(values) == 1:
                    values = list(values)[0]
                kwargs[name] = values
        if proto is not None:
            kwargs['proto'] = proto
        if impl is not None:
            kwargs['impl'] = impl
        if 'name' not in kwargs:
            kwargs['name'] = util_code_name
        if 'file' not in kwargs and from_file:
            kwargs['file'] = from_file
        return cls(**kwargs)

    @classmethod
    def load_cached(cls, utility_code_name, from_file, __cache={}):
        """
        Calls .load(), but using a per-type cache based on utility name and file name.
        """
        key = (utility_code_name, from_file, cls)
        try:
            return __cache[key]
        except KeyError:
            pass
        code = __cache[key] = cls.load(utility_code_name, from_file)
        return code

    @classmethod
    def load_as_string(cls, util_code_name, from_file, **kwargs):
        """
        Load a utility code as a string. Returns (proto, implementation)
        """
        util = cls.load(util_code_name, from_file, **kwargs)
        proto, impl = (util.proto, util.impl)
        return (util.format_code(proto), util.format_code(impl))

    def format_code(self, code_string, replace_empty_lines=re.compile('\\n\\n+').sub):
        """
        Format a code section for output.
        """
        if code_string:
            code_string = replace_empty_lines('\n', code_string.strip()) + '\n\n'
        return code_string

    def __repr__(self):
        return '<%s(%s)>' % (type(self).__name__, self.name)

    def get_tree(self, **kwargs):
        return None

    def __deepcopy__(self, memodict=None):
        return self