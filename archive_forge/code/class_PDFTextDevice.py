from typing import (
from pdfminer.psparser import PSLiteral
from . import utils
from .pdfcolor import PDFColorSpace
from .pdffont import PDFFont
from .pdffont import PDFUnicodeNotDefined
from .pdfpage import PDFPage
from .pdftypes import PDFStream
from .utils import Matrix, Point, Rect, PathSegment
class PDFTextDevice(PDFDevice):

    def render_string(self, textstate: 'PDFTextState', seq: PDFTextSeq, ncs: PDFColorSpace, graphicstate: 'PDFGraphicState') -> None:
        assert self.ctm is not None
        matrix = utils.mult_matrix(textstate.matrix, self.ctm)
        font = textstate.font
        fontsize = textstate.fontsize
        scaling = textstate.scaling * 0.01
        charspace = textstate.charspace * scaling
        wordspace = textstate.wordspace * scaling
        rise = textstate.rise
        assert font is not None
        if font.is_multibyte():
            wordspace = 0
        dxscale = 0.001 * fontsize * scaling
        if font.is_vertical():
            textstate.linematrix = self.render_string_vertical(seq, matrix, textstate.linematrix, font, fontsize, scaling, charspace, wordspace, rise, dxscale, ncs, graphicstate)
        else:
            textstate.linematrix = self.render_string_horizontal(seq, matrix, textstate.linematrix, font, fontsize, scaling, charspace, wordspace, rise, dxscale, ncs, graphicstate)

    def render_string_horizontal(self, seq: PDFTextSeq, matrix: Matrix, pos: Point, font: PDFFont, fontsize: float, scaling: float, charspace: float, wordspace: float, rise: float, dxscale: float, ncs: PDFColorSpace, graphicstate: 'PDFGraphicState') -> Point:
        x, y = pos
        needcharspace = False
        for obj in seq:
            if isinstance(obj, (int, float)):
                x -= obj * dxscale
                needcharspace = True
            else:
                for cid in font.decode(obj):
                    if needcharspace:
                        x += charspace
                    x += self.render_char(utils.translate_matrix(matrix, (x, y)), font, fontsize, scaling, rise, cid, ncs, graphicstate)
                    if cid == 32 and wordspace:
                        x += wordspace
                    needcharspace = True
        return (x, y)

    def render_string_vertical(self, seq: PDFTextSeq, matrix: Matrix, pos: Point, font: PDFFont, fontsize: float, scaling: float, charspace: float, wordspace: float, rise: float, dxscale: float, ncs: PDFColorSpace, graphicstate: 'PDFGraphicState') -> Point:
        x, y = pos
        needcharspace = False
        for obj in seq:
            if isinstance(obj, (int, float)):
                y -= obj * dxscale
                needcharspace = True
            else:
                for cid in font.decode(obj):
                    if needcharspace:
                        y += charspace
                    y += self.render_char(utils.translate_matrix(matrix, (x, y)), font, fontsize, scaling, rise, cid, ncs, graphicstate)
                    if cid == 32 and wordspace:
                        y += wordspace
                    needcharspace = True
        return (x, y)

    def render_char(self, matrix: Matrix, font: PDFFont, fontsize: float, scaling: float, rise: float, cid: int, ncs: PDFColorSpace, graphicstate: 'PDFGraphicState') -> float:
        return 0