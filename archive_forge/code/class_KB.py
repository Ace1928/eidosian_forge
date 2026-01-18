import xcffib
import struct
import io
class KB:
    KeyClickPercent = 1 << 0
    BellPercent = 1 << 1
    BellPitch = 1 << 2
    BellDuration = 1 << 3
    Led = 1 << 4
    LedMode = 1 << 5
    Key = 1 << 6
    AutoRepeatMode = 1 << 7