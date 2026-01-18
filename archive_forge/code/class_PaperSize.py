from typing import NamedTuple
class PaperSize:
    """(width, height) of the paper in portrait mode in pixels at 72 ppi."""
    A0 = Dimensions(2384, 3370)
    A1 = Dimensions(1684, 2384)
    A2 = Dimensions(1191, 1684)
    A3 = Dimensions(842, 1191)
    A4 = Dimensions(595, 842)
    A5 = Dimensions(420, 595)
    A6 = Dimensions(298, 420)
    A7 = Dimensions(210, 298)
    A8 = Dimensions(147, 210)
    C4 = Dimensions(649, 918)