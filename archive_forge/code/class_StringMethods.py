from __future__ import annotations
import re
from typing import TYPE_CHECKING
import numpy as np
import pandas
from modin.logging import ClassLogger
from modin.utils import _inherit_docstrings
@_inherit_docstrings(pandas.core.strings.accessor.StringMethods)
class StringMethods(ClassLogger):
    _series: Series
    _query_compiler: BaseQueryCompiler

    def __init__(self, data: Series):
        self._series = data
        self._query_compiler = data._query_compiler

    @pandas.util.cache_readonly
    def _Series(self):
        from .series import Series
        return Series

    def casefold(self):
        return self._Series(query_compiler=self._query_compiler.str_casefold())

    def cat(self, others=None, sep=None, na_rep=None, join='left'):
        if isinstance(others, self._Series):
            others = others._to_pandas()
        compiler_result = self._query_compiler.str_cat(others=others, sep=sep, na_rep=na_rep, join=join)
        return compiler_result.to_pandas().squeeze() if others is None else self._Series(query_compiler=compiler_result)

    def decode(self, encoding, errors='strict'):
        return self._Series(query_compiler=self._query_compiler.str_decode(encoding, errors))

    def split(self, pat=None, *, n=-1, expand=False, regex=None):
        if expand:
            from .dataframe import DataFrame
            return DataFrame(query_compiler=self._query_compiler.str_split(pat=pat, n=n, expand=True, regex=regex))
        else:
            return self._Series(query_compiler=self._query_compiler.str_split(pat=pat, n=n, expand=expand, regex=regex))

    def rsplit(self, pat=None, *, n=-1, expand=False):
        if not pat and pat is not None:
            raise ValueError('rsplit() requires a non-empty pattern match.')
        if expand:
            from .dataframe import DataFrame
            return DataFrame(query_compiler=self._query_compiler.str_rsplit(pat=pat, n=n, expand=True))
        else:
            return self._Series(query_compiler=self._query_compiler.str_rsplit(pat=pat, n=n, expand=expand))

    def get(self, i):
        return self._Series(query_compiler=self._query_compiler.str_get(i))

    def join(self, sep):
        if sep is None:
            raise AttributeError("'NoneType' object has no attribute 'join'")
        return self._Series(query_compiler=self._query_compiler.str_join(sep))

    def get_dummies(self, sep='|'):
        return self._Series(query_compiler=self._query_compiler.str_get_dummies(sep))

    def contains(self, pat, case=True, flags=0, na=None, regex=True):
        if pat is None and (not case):
            raise AttributeError("'NoneType' object has no attribute 'upper'")
        return self._Series(query_compiler=self._query_compiler.str_contains(pat, case=case, flags=flags, na=na, regex=regex))

    def replace(self, pat, repl, n=-1, case=None, flags=0, regex=False):
        if not (isinstance(repl, str) or callable(repl)):
            raise TypeError('repl must be a string or callable')
        return self._Series(query_compiler=self._query_compiler.str_replace(pat, repl, n=n, case=case, flags=flags, regex=regex))

    def pad(self, width, side='left', fillchar=' '):
        if len(fillchar) != 1:
            raise TypeError('fillchar must be a character, not str')
        return self._Series(query_compiler=self._query_compiler.str_pad(width, side=side, fillchar=fillchar))

    def center(self, width, fillchar=' '):
        if len(fillchar) != 1:
            raise TypeError('fillchar must be a character, not str')
        return self._Series(query_compiler=self._query_compiler.str_center(width, fillchar=fillchar))

    def ljust(self, width, fillchar=' '):
        if len(fillchar) != 1:
            raise TypeError('fillchar must be a character, not str')
        return self._Series(query_compiler=self._query_compiler.str_ljust(width, fillchar=fillchar))

    def rjust(self, width, fillchar=' '):
        if len(fillchar) != 1:
            raise TypeError('fillchar must be a character, not str')
        return self._Series(query_compiler=self._query_compiler.str_rjust(width, fillchar=fillchar))

    def zfill(self, width):
        return self._Series(query_compiler=self._query_compiler.str_zfill(width))

    def wrap(self, width, **kwargs):
        if width <= 0:
            raise ValueError('invalid width {} (must be > 0)'.format(width))
        return self._Series(query_compiler=self._query_compiler.str_wrap(width, **kwargs))

    def slice(self, start=None, stop=None, step=None):
        if step == 0:
            raise ValueError('slice step cannot be zero')
        return self._Series(query_compiler=self._query_compiler.str_slice(start=start, stop=stop, step=step))

    def slice_replace(self, start=None, stop=None, repl=None):
        return self._Series(query_compiler=self._query_compiler.str_slice_replace(start=start, stop=stop, repl=repl))

    def count(self, pat, flags=0):
        if not isinstance(pat, (str, re.Pattern)):
            raise TypeError('first argument must be string or compiled pattern')
        return self._Series(query_compiler=self._query_compiler.str_count(pat, flags=flags))

    def startswith(self, pat, na=None):
        return self._Series(query_compiler=self._query_compiler.str_startswith(pat, na=na))

    def encode(self, encoding, errors='strict'):
        return self._Series(query_compiler=self._query_compiler.str_encode(encoding, errors))

    def endswith(self, pat, na=None):
        return self._Series(query_compiler=self._query_compiler.str_endswith(pat, na=na))

    def findall(self, pat, flags=0):
        if not isinstance(pat, (str, re.Pattern)):
            raise TypeError('first argument must be string or compiled pattern')
        return self._Series(query_compiler=self._query_compiler.str_findall(pat, flags=flags))

    def fullmatch(self, pat, case=True, flags=0, na=None):
        if not isinstance(pat, (str, re.Pattern)):
            raise TypeError('first argument must be string or compiled pattern')
        return self._Series(query_compiler=self._query_compiler.str_fullmatch(pat, case=case, flags=flags, na=na))

    def match(self, pat, case=True, flags=0, na=None):
        if not isinstance(pat, (str, re.Pattern)):
            raise TypeError('first argument must be string or compiled pattern')
        return self._Series(query_compiler=self._query_compiler.str_match(pat, case=case, flags=flags, na=na))

    def extract(self, pat, flags=0, expand=True):
        query_compiler = self._query_compiler.str_extract(pat, flags=flags, expand=expand)
        from .dataframe import DataFrame
        return DataFrame(query_compiler=query_compiler) if expand or re.compile(pat).groups > 1 else self._Series(query_compiler=query_compiler)

    def extractall(self, pat, flags=0):
        return self._Series(query_compiler=self._query_compiler.str_extractall(pat, flags))

    def len(self):
        return self._Series(query_compiler=self._query_compiler.str_len())

    def strip(self, to_strip=None):
        return self._Series(query_compiler=self._query_compiler.str_strip(to_strip=to_strip))

    def rstrip(self, to_strip=None):
        return self._Series(query_compiler=self._query_compiler.str_rstrip(to_strip=to_strip))

    def lstrip(self, to_strip=None):
        return self._Series(query_compiler=self._query_compiler.str_lstrip(to_strip=to_strip))

    def partition(self, sep=' ', expand=True):
        if sep is not None and len(sep) == 0:
            raise ValueError('empty separator')
        from .dataframe import DataFrame
        return (DataFrame if expand else self._Series)(query_compiler=self._query_compiler.str_partition(sep=sep, expand=expand))

    def removeprefix(self, prefix):
        return self._Series(query_compiler=self._query_compiler.str_removeprefix(prefix))

    def removesuffix(self, suffix):
        return self._Series(query_compiler=self._query_compiler.str_removesuffix(suffix))

    def repeat(self, repeats):
        return self._Series(query_compiler=self._query_compiler.str_repeat(repeats))

    def rpartition(self, sep=' ', expand=True):
        if sep is not None and len(sep) == 0:
            raise ValueError('empty separator')
        from .dataframe import DataFrame
        return (DataFrame if expand else self._Series)(query_compiler=self._query_compiler.str_rpartition(sep=sep, expand=expand))

    def lower(self):
        return self._Series(query_compiler=self._query_compiler.str_lower())

    def upper(self):
        return self._Series(query_compiler=self._query_compiler.str_upper())

    def title(self):
        return self._Series(query_compiler=self._query_compiler.str_title())

    def find(self, sub, start=0, end=None):
        if not isinstance(sub, str):
            raise TypeError('expected a string object, not {0}'.format(type(sub).__name__))
        return self._Series(query_compiler=self._query_compiler.str_find(sub, start=start, end=end))

    def rfind(self, sub, start=0, end=None):
        if not isinstance(sub, str):
            raise TypeError('expected a string object, not {0}'.format(type(sub).__name__))
        return self._Series(query_compiler=self._query_compiler.str_rfind(sub, start=start, end=end))

    def index(self, sub, start=0, end=None):
        if not isinstance(sub, str):
            raise TypeError('expected a string object, not {0}'.format(type(sub).__name__))
        return self._Series(query_compiler=self._query_compiler.str_index(sub, start=start, end=end))

    def rindex(self, sub, start=0, end=None):
        if not isinstance(sub, str):
            raise TypeError('expected a string object, not {0}'.format(type(sub).__name__))
        return self._Series(query_compiler=self._query_compiler.str_rindex(sub, start=start, end=end))

    def capitalize(self):
        return self._Series(query_compiler=self._query_compiler.str_capitalize())

    def swapcase(self):
        return self._Series(query_compiler=self._query_compiler.str_swapcase())

    def normalize(self, form):
        return self._Series(query_compiler=self._query_compiler.str_normalize(form))

    def translate(self, table):
        return self._Series(query_compiler=self._query_compiler.str_translate(table))

    def isalnum(self):
        return self._Series(query_compiler=self._query_compiler.str_isalnum())

    def isalpha(self):
        return self._Series(query_compiler=self._query_compiler.str_isalpha())

    def isdigit(self):
        return self._Series(query_compiler=self._query_compiler.str_isdigit())

    def isspace(self):
        return self._Series(query_compiler=self._query_compiler.str_isspace())

    def islower(self):
        return self._Series(query_compiler=self._query_compiler.str_islower())

    def isupper(self):
        return self._Series(query_compiler=self._query_compiler.str_isupper())

    def istitle(self):
        return self._Series(query_compiler=self._query_compiler.str_istitle())

    def isnumeric(self):
        return self._Series(query_compiler=self._query_compiler.str_isnumeric())

    def isdecimal(self):
        return self._Series(query_compiler=self._query_compiler.str_isdecimal())

    def __getitem__(self, key):
        return self._Series(query_compiler=self._query_compiler.str___getitem__(key))

    def _default_to_pandas(self, op, *args, **kwargs):
        """
        Convert `self` to pandas type and call a pandas str.`op` on it.

        Parameters
        ----------
        op : str
            Name of pandas function.
        *args : list
            Additional positional arguments to be passed in `op`.
        **kwargs : dict
            Additional keywords arguments to be passed in `op`.

        Returns
        -------
        object
            Result of operation.
        """
        return self._series._default_to_pandas(lambda series: op(series.str, *args, **kwargs))