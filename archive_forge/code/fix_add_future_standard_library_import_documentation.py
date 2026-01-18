from lib2to3 import fixer_base
from libfuturize.fixer_util import touch_import_top

For the ``future`` package.

Adds this import line:

    from future import standard_library

after any __future__ imports but before any other imports. Doesn't actually
change the imports to Py3 style.
