import array
import math
import os
import re
from PIL import Image
from rdkit.Chem.Draw.canvasbase import CanvasBase
def _addCanvasText1(self, text, pos, font, color=(0, 0, 0), **kwargs):
    if font.weight == 'bold':
        weight = cairo.FONT_WEIGHT_BOLD
    else:
        weight = cairo.FONT_WEIGHT_NORMAL
    self.ctx.select_font_face(font.face, cairo.FONT_SLANT_NORMAL, weight)
    text = scriptPattern.sub('', text)
    self.ctx.set_font_size(font.size)
    w, h = self.ctx.text_extents(text)[2:4]
    bw, bh = (w + h * 0.4, h * 1.4)
    offset = w * pos[2]
    dPos = (pos[0] - w / 2.0 + offset, pos[1] + h / 2.0)
    self.ctx.set_source_rgb(*color)
    self.ctx.move_to(*dPos)
    self.ctx.show_text(text)
    if 0:
        self.ctx.move_to(dPos[0], dPos[1])
        self.ctx.line_to(dPos[0] + bw, dPos[1])
        self.ctx.line_to(dPos[0] + bw, dPos[1] - bh)
        self.ctx.line_to(dPos[0], dPos[1] - bh)
        self.ctx.line_to(dPos[0], dPos[1])
        self.ctx.close_path()
        self.ctx.stroke()
    return (bw, bh, offset)