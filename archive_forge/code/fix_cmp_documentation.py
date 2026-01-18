from __future__ import unicode_literals
from lib2to3 import fixer_base
from libfuturize.fixer_util import touch_import_top

Fixer for the cmp() function on Py2, which was removed in Py3.

Adds this import line::

    from past.builtins import cmp

if cmp() is called in the code.
