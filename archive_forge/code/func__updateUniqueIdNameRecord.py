from contextlib import contextmanager
from copy import deepcopy
from enum import IntEnum
import re
def _updateUniqueIdNameRecord(varfont, nameIDs, platform):
    nametable = varfont['name']
    currentRecord = nametable.getName(NameID.UNIQUE_FONT_IDENTIFIER, *platform)
    if not currentRecord:
        return None
    for nameID in (NameID.FULL_FONT_NAME, NameID.POSTSCRIPT_NAME):
        nameRecord = nametable.getName(nameID, *platform)
        if not nameRecord:
            continue
        if nameRecord.toUnicode() in currentRecord.toUnicode():
            return currentRecord.toUnicode().replace(nameRecord.toUnicode(), nameIDs[nameRecord.nameID])
    fontVersion = _fontVersion(varfont, platform)
    achVendID = varfont['OS/2'].achVendID
    vendor = re.sub('[^\\x00-\\x7F]', '', achVendID).strip()
    psName = nameIDs[NameID.POSTSCRIPT_NAME]
    return f'{fontVersion};{vendor};{psName}'