import os
import re
import string
import textwrap
import warnings
from string import Formatter
from pathlib import Path
from typing import List, Dict, Tuple, Optional, cast
class SList(list):
    """List derivative with a special access attributes.

    These are normal lists, but with the special attributes:

    * .l (or .list) : value as list (the list itself).
    * .n (or .nlstr): value as a string, joined on newlines.
    * .s (or .spstr): value as a string, joined on spaces.
    * .p (or .paths): list of path objects (requires path.py package)

    Any values which require transformations are computed only once and
    cached."""

    def get_list(self):
        return self
    l = list = property(get_list)

    def get_spstr(self):
        try:
            return self.__spstr
        except AttributeError:
            self.__spstr = ' '.join(self)
            return self.__spstr
    s = spstr = property(get_spstr)

    def get_nlstr(self):
        try:
            return self.__nlstr
        except AttributeError:
            self.__nlstr = '\n'.join(self)
            return self.__nlstr
    n = nlstr = property(get_nlstr)

    def get_paths(self):
        try:
            return self.__paths
        except AttributeError:
            self.__paths = [Path(p) for p in self if os.path.exists(p)]
            return self.__paths
    p = paths = property(get_paths)

    def grep(self, pattern, prune=False, field=None):
        """ Return all strings matching 'pattern' (a regex or callable)

        This is case-insensitive. If prune is true, return all items
        NOT matching the pattern.

        If field is specified, the match must occur in the specified
        whitespace-separated field.

        Examples::

            a.grep( lambda x: x.startswith('C') )
            a.grep('Cha.*log', prune=1)
            a.grep('chm', field=-1)
        """

        def match_target(s):
            if field is None:
                return s
            parts = s.split()
            try:
                tgt = parts[field]
                return tgt
            except IndexError:
                return ''
        if isinstance(pattern, str):
            pred = lambda x: re.search(pattern, x, re.IGNORECASE)
        else:
            pred = pattern
        if not prune:
            return SList([el for el in self if pred(match_target(el))])
        else:
            return SList([el for el in self if not pred(match_target(el))])

    def fields(self, *fields):
        """ Collect whitespace-separated fields from string list

        Allows quick awk-like usage of string lists.

        Example data (in var a, created by 'a = !ls -l')::

            -rwxrwxrwx  1 ville None      18 Dec 14  2006 ChangeLog
            drwxrwxrwx+ 6 ville None       0 Oct 24 18:05 IPython

        * ``a.fields(0)`` is ``['-rwxrwxrwx', 'drwxrwxrwx+']``
        * ``a.fields(1,0)`` is ``['1 -rwxrwxrwx', '6 drwxrwxrwx+']``
          (note the joining by space).
        * ``a.fields(-1)`` is ``['ChangeLog', 'IPython']``

        IndexErrors are ignored.

        Without args, fields() just split()'s the strings.
        """
        if len(fields) == 0:
            return [el.split() for el in self]
        res = SList()
        for el in [f.split() for f in self]:
            lineparts = []
            for fd in fields:
                try:
                    lineparts.append(el[fd])
                except IndexError:
                    pass
            if lineparts:
                res.append(' '.join(lineparts))
        return res

    def sort(self, field=None, nums=False):
        """ sort by specified fields (see fields())

        Example::

            a.sort(1, nums = True)

        Sorts a by second field, in numerical order (so that 21 > 3)

        """
        if field is not None:
            dsu = [[SList([line]).fields(field), line] for line in self]
        else:
            dsu = [[line, line] for line in self]
        if nums:
            for i in range(len(dsu)):
                numstr = ''.join([ch for ch in dsu[i][0] if ch.isdigit()])
                try:
                    n = int(numstr)
                except ValueError:
                    n = 0
                dsu[i][0] = n
        dsu.sort()
        return SList([t[1] for t in dsu])