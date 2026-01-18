from contextlib import contextmanager
from copy import deepcopy
from enum import IntEnum
import re
def _updateNameRecords(varfont, axisValues):
    nametable = varfont['name']
    stat = varfont['STAT'].table
    axisValueNameIDs = [a.ValueNameID for a in axisValues]
    ribbiNameIDs = [n for n in axisValueNameIDs if _isRibbi(nametable, n)]
    nonRibbiNameIDs = [n for n in axisValueNameIDs if n not in ribbiNameIDs]
    elidedNameID = stat.ElidedFallbackNameID
    elidedNameIsRibbi = _isRibbi(nametable, elidedNameID)
    getName = nametable.getName
    platforms = set(((r.platformID, r.platEncID, r.langID) for r in nametable.names))
    for platform in platforms:
        if not all((getName(i, *platform) for i in (1, 2, elidedNameID))):
            continue
        subFamilyName = ' '.join((getName(n, *platform).toUnicode() for n in ribbiNameIDs))
        if nonRibbiNameIDs:
            typoSubFamilyName = ' '.join((getName(n, *platform).toUnicode() for n in axisValueNameIDs))
        else:
            typoSubFamilyName = None
        if not typoSubFamilyName and (not subFamilyName):
            if elidedNameIsRibbi:
                subFamilyName = getName(elidedNameID, *platform).toUnicode()
            else:
                typoSubFamilyName = getName(elidedNameID, *platform).toUnicode()
        familyNameSuffix = ' '.join((getName(n, *platform).toUnicode() for n in nonRibbiNameIDs))
        _updateNameTableStyleRecords(varfont, familyNameSuffix, subFamilyName, typoSubFamilyName, *platform)