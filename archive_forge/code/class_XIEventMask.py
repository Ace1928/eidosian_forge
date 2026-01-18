import xcffib
import struct
import io
from . import xfixes
from . import xproto
class XIEventMask:
    DeviceChanged = 1 << 1
    KeyPress = 1 << 2
    KeyRelease = 1 << 3
    ButtonPress = 1 << 4
    ButtonRelease = 1 << 5
    Motion = 1 << 6
    Enter = 1 << 7
    Leave = 1 << 8
    FocusIn = 1 << 9
    FocusOut = 1 << 10
    Hierarchy = 1 << 11
    Property = 1 << 12
    RawKeyPress = 1 << 13
    RawKeyRelease = 1 << 14
    RawButtonPress = 1 << 15
    RawButtonRelease = 1 << 16
    RawMotion = 1 << 17
    TouchBegin = 1 << 18
    TouchUpdate = 1 << 19
    TouchEnd = 1 << 20
    TouchOwnership = 1 << 21
    RawTouchBegin = 1 << 22
    RawTouchUpdate = 1 << 23
    RawTouchEnd = 1 << 24
    BarrierHit = 1 << 25
    BarrierLeave = 1 << 26