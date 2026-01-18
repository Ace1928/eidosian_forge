import xcffib
import struct
import io
class ModeFlag:
    Positive_HSync = 1 << 0
    Negative_HSync = 1 << 1
    Positive_VSync = 1 << 2
    Negative_VSync = 1 << 3
    Interlace = 1 << 4
    Composite_Sync = 1 << 5
    Positive_CSync = 1 << 6
    Negative_CSync = 1 << 7
    HSkew = 1 << 8
    Broadcast = 1 << 9
    Pixmux = 1 << 10
    Double_Clock = 1 << 11
    Half_Clock = 1 << 12