from fontTools.misc.textTools import bytechr, bytesjoin, byteord
def _encryptChar(plain, R):
    plain = byteord(plain)
    cipher = (plain ^ R >> 8) & 255
    R = (cipher + R) * 52845 + 22719 & 65535
    return (bytechr(cipher), R)