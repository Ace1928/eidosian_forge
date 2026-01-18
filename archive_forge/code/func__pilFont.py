from rdkit.sping.pid import *
import math
import os
def _pilFont(font):
    if font.face:
        face = font.face
    else:
        face = 'times'
    size = _closestSize(font.size)
    if isinstance(face, str):
        try:
            pilfont = ImageFont.load_path(_pilFontPath(face, size, font.bold))
        except Exception:
            return 0
    else:
        for item in font.face:
            pilfont = None
            try:
                pilfont = ImageFont.load_path(_pilFontPath(item, size, font.bold))
                break
            except Exception:
                pass
        if pilfont is None:
            return 0
    return pilfont