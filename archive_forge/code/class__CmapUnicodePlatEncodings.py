from fontTools.merge.unicode import is_Default_Ignorable
from fontTools.pens.recordingPen import DecomposingRecordingPen
import logging
class _CmapUnicodePlatEncodings:
    BMP = {(4, 3, 1), (4, 0, 3), (4, 0, 4), (4, 0, 6)}
    FullRepertoire = {(12, 3, 10), (12, 0, 4), (12, 0, 6)}