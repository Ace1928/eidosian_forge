from os.path import dirname, join, exists, sep
from lib2to3.fixes.fix_import import FixImport
from lib2to3.fixer_util import FromImport, syms
from lib2to3.fixes.fix_import import traverse_imports
from libfuturize.fixer_util import future_import

        Like the corresponding method in the base class, but this also
        supports Cython modules.
        