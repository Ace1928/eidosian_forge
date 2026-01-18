from lib2to3 import fixer_base
from lib2to3.fixer_util import Name

Fixer for os.getcwd() -> os.getcwdu().
Also warns about "from os import getcwd", suggesting the above form.
