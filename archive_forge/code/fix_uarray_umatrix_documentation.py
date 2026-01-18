from lib2to3.fixer_base import BaseFix
from lib2to3.fixer_util import String, ArgList, Comma, syms

Fixer for lib2to3.

Transforms uarray(tuple) into uarray(nominal_values, std_devs) and
uarray(single_arg) into uarray(*single_arg).

(c) 2013 by Eric O. LEBIGOT (EOL).
