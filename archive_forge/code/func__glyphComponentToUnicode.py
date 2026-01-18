from fontTools.misc.textTools import tostr
import re
def _glyphComponentToUnicode(component, isZapfDingbats):
    dingbat = _zapfDingbatsToUnicode(component) if isZapfDingbats else None
    if dingbat:
        return dingbat
    uchars = LEGACY_AGL2UV.get(component)
    if uchars:
        return ''.join(map(chr, uchars))
    uni = _uniToUnicode(component)
    if uni:
        return uni
    uni = _uToUnicode(component)
    if uni:
        return uni
    return ''