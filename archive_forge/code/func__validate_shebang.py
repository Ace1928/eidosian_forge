import os
import re
from stat import ST_MODE
from distutils import sysconfig
from ..core import Command
from .._modified import newer
from ..util import convert_path
from distutils._log import log
import tokenize
@staticmethod
def _validate_shebang(shebang, encoding):
    try:
        shebang.encode('utf-8')
    except UnicodeEncodeError:
        raise ValueError('The shebang ({!r}) is not encodable to utf-8'.format(shebang))
    try:
        shebang.encode(encoding)
    except UnicodeEncodeError:
        raise ValueError('The shebang ({!r}) is not encodable to the script encoding ({})'.format(shebang, encoding))