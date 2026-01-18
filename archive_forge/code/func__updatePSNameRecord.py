from contextlib import contextmanager
from copy import deepcopy
from enum import IntEnum
import re
def _updatePSNameRecord(varfont, familyName, styleName, platform):
    nametable = varfont['name']
    family_prefix = nametable.getName(NameID.VARIATIONS_POSTSCRIPT_NAME_PREFIX, *platform)
    if family_prefix:
        family_prefix = family_prefix.toUnicode()
    else:
        family_prefix = familyName
    psName = f'{family_prefix}-{styleName}'
    psName = re.sub('[^A-Za-z0-9-]', '', psName)
    if len(psName) > 127:
        return f'{psName[:124]}...'
    return psName