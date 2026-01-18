from rdkit.sping.pid import *
import math
import os
def _matchingFontPath(font):
    if font.face:
        face = font.face
    else:
        face = 'times'
    size = _closestSize(font.size)
    if isinstance(face, str):
        path = _pilFontPath(face, size, font.bold)
        path = path.split(os.sep)[-1]
        if path in _widthmaps.keys():
            return path
    else:
        for item in font.face:
            path = _pilFontPath(item, size, font.bold)
            path = path.split(os.sep)[-1]
            if path in _widthmaps.keys():
                return path
    path = _pilFontPath('courier', size, font.bold)
    return path.split(os.sep)[-1]