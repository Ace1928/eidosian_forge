from contextlib import contextmanager
from copy import deepcopy
from enum import IntEnum
import re
def _updateNameTableStyleRecords(varfont, familyNameSuffix, subFamilyName, typoSubFamilyName, platformID=3, platEncID=1, langID=1033):
    nametable = varfont['name']
    platform = (platformID, platEncID, langID)
    currentFamilyName = nametable.getName(NameID.TYPOGRAPHIC_FAMILY_NAME, *platform) or nametable.getName(NameID.FAMILY_NAME, *platform)
    currentStyleName = nametable.getName(NameID.TYPOGRAPHIC_SUBFAMILY_NAME, *platform) or nametable.getName(NameID.SUBFAMILY_NAME, *platform)
    if not all([currentFamilyName, currentStyleName]):
        raise ValueError(f'Missing required NameIDs 1 and 2 for platform {platform}')
    currentFamilyName = currentFamilyName.toUnicode()
    currentStyleName = currentStyleName.toUnicode()
    nameIDs = {NameID.FAMILY_NAME: currentFamilyName, NameID.SUBFAMILY_NAME: subFamilyName or 'Regular'}
    if typoSubFamilyName:
        nameIDs[NameID.FAMILY_NAME] = f'{currentFamilyName} {familyNameSuffix}'.strip()
        nameIDs[NameID.TYPOGRAPHIC_FAMILY_NAME] = currentFamilyName
        nameIDs[NameID.TYPOGRAPHIC_SUBFAMILY_NAME] = typoSubFamilyName
    else:
        for nameID in (NameID.TYPOGRAPHIC_FAMILY_NAME, NameID.TYPOGRAPHIC_SUBFAMILY_NAME):
            nametable.removeNames(nameID=nameID)
    newFamilyName = nameIDs.get(NameID.TYPOGRAPHIC_FAMILY_NAME) or nameIDs[NameID.FAMILY_NAME]
    newStyleName = nameIDs.get(NameID.TYPOGRAPHIC_SUBFAMILY_NAME) or nameIDs[NameID.SUBFAMILY_NAME]
    nameIDs[NameID.FULL_FONT_NAME] = f'{newFamilyName} {newStyleName}'
    nameIDs[NameID.POSTSCRIPT_NAME] = _updatePSNameRecord(varfont, newFamilyName, newStyleName, platform)
    uniqueID = _updateUniqueIdNameRecord(varfont, nameIDs, platform)
    if uniqueID:
        nameIDs[NameID.UNIQUE_FONT_IDENTIFIER] = uniqueID
    for nameID, string in nameIDs.items():
        assert string, nameID
        nametable.setName(string, nameID, *platform)
    if 'fvar' not in varfont:
        nametable.removeNames(NameID.VARIATIONS_POSTSCRIPT_NAME_PREFIX)