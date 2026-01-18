import re, os, stat, io
from xdg.Exceptions import (ParsingError, DuplicateGroupError, NoGroupError,
import xdg.Locale
from xdg.util import u
def __getBoolean(self, boolean):
    if boolean == 1 or boolean == 'true' or boolean == 'True':
        return True
    elif boolean == 0 or boolean == 'false' or boolean == 'False':
        return False
    return False