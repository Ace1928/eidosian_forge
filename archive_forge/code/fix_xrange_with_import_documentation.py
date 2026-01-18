from lib2to3.fixes.fix_xrange import FixXrange
from libfuturize.fixer_util import touch_import_top

For the ``future`` package.

Turns any xrange calls into range calls and adds this import line:

    from builtins import range

at the top.
