from fontTools.misc.textTools import tostr
import re
def _zapfDingbatsToUnicode(glyph):
    """Helper for toUnicode()."""
    if len(glyph) < 2 or glyph[0] != 'a':
        return None
    try:
        gid = int(glyph[1:])
    except ValueError:
        return None
    if gid < 0 or gid >= len(_AGL_ZAPF_DINGBATS):
        return None
    uchar = _AGL_ZAPF_DINGBATS[gid]
    return uchar if uchar != ' ' else None