import xcffib
import struct
import io
class GC:
    Function = 1 << 0
    PlaneMask = 1 << 1
    Foreground = 1 << 2
    Background = 1 << 3
    LineWidth = 1 << 4
    LineStyle = 1 << 5
    CapStyle = 1 << 6
    JoinStyle = 1 << 7
    FillStyle = 1 << 8
    FillRule = 1 << 9
    Tile = 1 << 10
    Stipple = 1 << 11
    TileStippleOriginX = 1 << 12
    TileStippleOriginY = 1 << 13
    Font = 1 << 14
    SubwindowMode = 1 << 15
    GraphicsExposures = 1 << 16
    ClipOriginX = 1 << 17
    ClipOriginY = 1 << 18
    ClipMask = 1 << 19
    DashOffset = 1 << 20
    DashList = 1 << 21
    ArcMode = 1 << 22