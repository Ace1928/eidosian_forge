from fontTools.misc.textTools import Tag, bytesjoin, strjoin
def _reverseString(s):
    s = list(s)
    s.reverse()
    return strjoin(s)