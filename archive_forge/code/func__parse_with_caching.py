import re
import sys
from pprint import pprint
def _parse_with_caching(self, check):
    if check in self._cache:
        fun_name, fun_args, fun_kwargs, default = self._cache[check]
        fun_args = list(fun_args)
        fun_kwargs = dict(fun_kwargs)
    else:
        fun_name, fun_args, fun_kwargs, default = self._parse_check(check)
        fun_kwargs = dict([(str(key), value) for key, value in list(fun_kwargs.items())])
        self._cache[check] = (fun_name, list(fun_args), dict(fun_kwargs), default)
    return (fun_name, fun_args, fun_kwargs, default)