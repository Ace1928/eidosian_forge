import xcffib
import struct
import io
def ChangeGC(self, gc, value_mask, value_list, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', gc, value_mask))
    if value_mask & GC.Function:
        function = value_list.pop(0)
        buf.write(struct.pack('=I', function))
    if value_mask & GC.PlaneMask:
        plane_mask = value_list.pop(0)
        buf.write(struct.pack('=I', plane_mask))
    if value_mask & GC.Foreground:
        foreground = value_list.pop(0)
        buf.write(struct.pack('=I', foreground))
    if value_mask & GC.Background:
        background = value_list.pop(0)
        buf.write(struct.pack('=I', background))
    if value_mask & GC.LineWidth:
        line_width = value_list.pop(0)
        buf.write(struct.pack('=I', line_width))
    if value_mask & GC.LineStyle:
        line_style = value_list.pop(0)
        buf.write(struct.pack('=I', line_style))
    if value_mask & GC.CapStyle:
        cap_style = value_list.pop(0)
        buf.write(struct.pack('=I', cap_style))
    if value_mask & GC.JoinStyle:
        join_style = value_list.pop(0)
        buf.write(struct.pack('=I', join_style))
    if value_mask & GC.FillStyle:
        fill_style = value_list.pop(0)
        buf.write(struct.pack('=I', fill_style))
    if value_mask & GC.FillRule:
        fill_rule = value_list.pop(0)
        buf.write(struct.pack('=I', fill_rule))
    if value_mask & GC.Tile:
        tile = value_list.pop(0)
        buf.write(struct.pack('=I', tile))
    if value_mask & GC.Stipple:
        stipple = value_list.pop(0)
        buf.write(struct.pack('=I', stipple))
    if value_mask & GC.TileStippleOriginX:
        tile_stipple_x_origin = value_list.pop(0)
        buf.write(struct.pack('=i', tile_stipple_x_origin))
    if value_mask & GC.TileStippleOriginY:
        tile_stipple_y_origin = value_list.pop(0)
        buf.write(struct.pack('=i', tile_stipple_y_origin))
    if value_mask & GC.Font:
        font = value_list.pop(0)
        buf.write(struct.pack('=I', font))
    if value_mask & GC.SubwindowMode:
        subwindow_mode = value_list.pop(0)
        buf.write(struct.pack('=I', subwindow_mode))
    if value_mask & GC.GraphicsExposures:
        graphics_exposures = value_list.pop(0)
        buf.write(struct.pack('=I', graphics_exposures))
    if value_mask & GC.ClipOriginX:
        clip_x_origin = value_list.pop(0)
        buf.write(struct.pack('=i', clip_x_origin))
    if value_mask & GC.ClipOriginY:
        clip_y_origin = value_list.pop(0)
        buf.write(struct.pack('=i', clip_y_origin))
    if value_mask & GC.ClipMask:
        clip_mask = value_list.pop(0)
        buf.write(struct.pack('=I', clip_mask))
    if value_mask & GC.DashOffset:
        dash_offset = value_list.pop(0)
        buf.write(struct.pack('=I', dash_offset))
    if value_mask & GC.DashList:
        dashes = value_list.pop(0)
        buf.write(struct.pack('=I', dashes))
    if value_mask & GC.ArcMode:
        arc_mode = value_list.pop(0)
        buf.write(struct.pack('=I', arc_mode))
    return self.send_request(56, buf, is_checked=is_checked)