import array
import math
import os
import re
from PIL import Image
from rdkit.Chem.Draw.canvasbase import CanvasBase
def _addCanvasText2(self, text, pos, font, color=(0, 0, 0), **kwargs):
    if font.weight == 'bold':
        weight = cairo.FONT_WEIGHT_BOLD
    else:
        weight = cairo.FONT_WEIGHT_NORMAL
    self.ctx.select_font_face(font.face, cairo.FONT_SLANT_NORMAL, weight)
    orientation = kwargs.get('orientation', 'E')
    plainText = scriptPattern.sub('', text)
    pangoCoeff = 0.8
    if have_cairocffi:
        measureLout = pangocairo.pango_cairo_create_layout(self.ctx._pointer)
        pango.pango_layout_set_alignment(measureLout, pango.PANGO_ALIGN_LEFT)
        pango.pango_layout_set_markup(measureLout, plainText.encode('latin1'), -1)
        lout = pangocairo.pango_cairo_create_layout(self.ctx._pointer)
        pango.pango_layout_set_alignment(lout, pango.PANGO_ALIGN_LEFT)
        pango.pango_layout_set_markup(lout, text.encode('latin1'), -1)
        fnt = pango.pango_font_description_new()
        pango.pango_font_description_set_family(fnt, font.face.encode('latin1'))
        pango.pango_font_description_set_size(fnt, int(round(font.size * pango.PANGO_SCALE * pangoCoeff)))
        pango.pango_layout_set_font_description(lout, fnt)
        pango.pango_layout_set_font_description(measureLout, fnt)
        pango.pango_font_description_free(fnt)
    else:
        cctx = pangocairo.CairoContext(self.ctx)
        measureLout = cctx.create_layout()
        measureLout.set_alignment(pango.ALIGN_LEFT)
        measureLout.set_markup(plainText)
        lout = cctx.create_layout()
        lout.set_alignment(pango.ALIGN_LEFT)
        lout.set_markup(text)
        fnt = pango.FontDescription('%s %d' % (font.face, font.size * pangoCoeff))
        lout.set_font_description(fnt)
        measureLout.set_font_description(fnt)
    if have_cairocffi:
        iext = ffi.new('PangoRectangle *')
        lext = ffi.new('PangoRectangle *')
        iext2 = ffi.new('PangoRectangle *')
        lext2 = ffi.new('PangoRectangle *')
        pango.pango_layout_get_pixel_extents(measureLout, iext, lext)
        pango.pango_layout_get_pixel_extents(lout, iext2, lext2)
        w = lext2.width - lext2.x
        h = lext.height - lext.y
    else:
        iext, lext = measureLout.get_pixel_extents()
        iext2, lext2 = lout.get_pixel_extents()
        w = lext2[2] - lext2[0]
        h = lext[3] - lext[1]
    pad = [h * 0.2, h * 0.3]
    if orientation == 'S':
        pad[1] *= 0.5
    bw, bh = (w + pad[0], h + pad[1])
    offset = w * pos[2]
    if 0:
        if orientation == 'W':
            dPos = (pos[0] - w + offset, pos[1] - h / 2.0)
        elif orientation == 'E':
            dPos = (pos[0] - w / 2 + offset, pos[1] - h / 2.0)
        else:
            dPos = (pos[0] - w / 2 + offset, pos[1] - h / 2.0)
        self.ctx.move_to(dPos[0], dPos[1])
    else:
        dPos = (pos[0] - w / 2.0 + offset, pos[1] - h / 2.0)
        self.ctx.move_to(dPos[0], dPos[1])
    self.ctx.set_source_rgb(*color)
    if have_cairocffi:
        pangocairo.pango_cairo_update_layout(self.ctx._pointer, lout)
        pangocairo.pango_cairo_show_layout(self.ctx._pointer, lout)
        gobject.g_object_unref(lout)
        gobject.g_object_unref(measureLout)
    else:
        cctx.update_layout(lout)
        cctx.show_layout(lout)
    if 0:
        self.ctx.move_to(dPos[0], dPos[1])
        self.ctx.line_to(dPos[0] + bw, dPos[1])
        self.ctx.line_to(dPos[0] + bw, dPos[1] + bh)
        self.ctx.line_to(dPos[0], dPos[1] + bh)
        self.ctx.line_to(dPos[0], dPos[1])
        self.ctx.close_path()
        self.ctx.stroke()
    return (bw, bh, offset)