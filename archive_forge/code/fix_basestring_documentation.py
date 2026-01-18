from lib2to3 import fixer_base
from libfuturize.fixer_util import touch_import_top

Fixer that adds ``from past.builtins import basestring`` if there is a
reference to ``basestring``
