from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import newTable
from fontTools.ttLib.ttVisitor import TTVisitor
from fontTools import ttLib
import fontTools.ttLib.tables.otTables as otTables
from fontTools.ttLib.tables import C_P_A_L_
from . import DefaultTable
import struct
import logging
def _makeMacName(name, nameID, language, font=None):
    """Create a NameRecord for Apple platforms

    'language' is an arbitrary IETF BCP 47 language identifier such
    as 'en', 'de-CH', 'de-AT-1901', or 'fa-Latn'. When possible, we
    create a Macintosh NameRecord that is understood by old applications
    (platform ID 1 and an old-style Macintosh language enum). If this
    is not possible, we create a Unicode NameRecord (platform ID 0)
    whose language points to the font’s 'ltag' table. The latter
    can encode any string in any language, but legacy applications
    might not recognize the format (in which case they will ignore
    those names).

    'font' should be the TTFont for which you want to create a name.
    If 'font' is None, we only return NameRecords for legacy Macintosh;
    in that case, the result will be None for names that need to
    be encoded with an 'ltag' table.

    See the section “The language identifier” in Apple’s specification:
    https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6name.html
    """
    macLang = _MAC_LANGUAGE_CODES.get(language.lower())
    macScript = _MAC_LANGUAGE_TO_SCRIPT.get(macLang)
    if macLang is not None and macScript is not None:
        encoding = getEncoding(1, macScript, macLang, default='ascii')
        try:
            _ = tobytes(name, encoding, errors='strict')
            return makeName(name, nameID, 1, macScript, macLang)
        except UnicodeEncodeError:
            pass
    if font is not None:
        ltag = font.tables.get('ltag')
        if ltag is None:
            ltag = font['ltag'] = newTable('ltag')
        return makeName(name, nameID, 0, 4, ltag.addTag(language))
    else:
        log.warning("cannot store language %s into 'ltag' table without having access to the TTFont object" % language)
        return None