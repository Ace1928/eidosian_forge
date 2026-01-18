import sys, os, ast, re, weakref, time, copy, math, types
import textwrap
def __rl_is_allowed_name__(self, name):
    """Check names if they are allowed.
		If ``allow_magic_methods is True`` names in `__allowed_magic_methods__`
		are additionally allowed although their names start with `_`.
		"""
    if isinstance(name, strTypes):
        if name in __rl_unsafe__ or (name.startswith('__') and name != '__' and (name not in self.allowed_magic_methods)):
            raise BadCode('unsafe access of %s' % name)