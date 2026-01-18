import codecs
from datetime import timezone
from datetime import datetime
from enum import Enum
from functools import total_ordering
from io import BytesIO
import itertools
import logging
import math
import os
import string
import struct
import sys
import time
import types
import warnings
import zlib
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import _api, _text_helpers, _type1font, cbook, dviread
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.figure import Figure
from matplotlib.font_manager import get_font, fontManager as _fontManager
from matplotlib._afm import AFM
from matplotlib.ft2font import (FIXED_WIDTH, ITALIC, LOAD_NO_SCALE,
from matplotlib.transforms import Affine2D, BboxBase
from matplotlib.path import Path
from matplotlib.dates import UTC
from matplotlib import _path
from . import _backend_pdf_ps
class RendererPdf(_backend_pdf_ps.RendererPDFPSBase):
    _afm_font_dir = cbook._get_data_path('fonts/pdfcorefonts')
    _use_afm_rc_name = 'pdf.use14corefonts'

    def __init__(self, file, image_dpi, height, width):
        super().__init__(width, height)
        self.file = file
        self.gc = self.new_gc()
        self.image_dpi = image_dpi

    def finalize(self):
        self.file.output(*self.gc.finalize())

    def check_gc(self, gc, fillcolor=None):
        orig_fill = getattr(gc, '_fillcolor', (0.0, 0.0, 0.0))
        gc._fillcolor = fillcolor
        orig_alphas = getattr(gc, '_effective_alphas', (1.0, 1.0))
        if gc.get_rgb() is None:
            gc.set_foreground((0, 0, 0, 0), isRGBA=True)
        if gc._forced_alpha:
            gc._effective_alphas = (gc._alpha, gc._alpha)
        elif fillcolor is None or len(fillcolor) < 4:
            gc._effective_alphas = (gc._rgb[3], 1.0)
        else:
            gc._effective_alphas = (gc._rgb[3], fillcolor[3])
        delta = self.gc.delta(gc)
        if delta:
            self.file.output(*delta)
        gc._fillcolor = orig_fill
        gc._effective_alphas = orig_alphas

    def get_image_magnification(self):
        return self.image_dpi / 72.0

    def draw_image(self, gc, x, y, im, transform=None):
        h, w = im.shape[:2]
        if w == 0 or h == 0:
            return
        if transform is None:
            gc.set_alpha(1.0)
        self.check_gc(gc)
        w = 72.0 * w / self.image_dpi
        h = 72.0 * h / self.image_dpi
        imob = self.file.imageObject(im)
        if transform is None:
            self.file.output(Op.gsave, w, 0, 0, h, x, y, Op.concat_matrix, imob, Op.use_xobject, Op.grestore)
        else:
            tr1, tr2, tr3, tr4, tr5, tr6 = transform.frozen().to_values()
            self.file.output(Op.gsave, 1, 0, 0, 1, x, y, Op.concat_matrix, tr1, tr2, tr3, tr4, tr5, tr6, Op.concat_matrix, imob, Op.use_xobject, Op.grestore)

    def draw_path(self, gc, path, transform, rgbFace=None):
        self.check_gc(gc, rgbFace)
        self.file.writePath(path, transform, rgbFace is None and gc.get_hatch_path() is None, gc.get_sketch_params())
        self.file.output(self.gc.paint())

    def draw_path_collection(self, gc, master_transform, paths, all_transforms, offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position):
        can_do_optimization = True
        facecolors = np.asarray(facecolors)
        edgecolors = np.asarray(edgecolors)
        if not len(facecolors):
            filled = False
            can_do_optimization = not gc.get_hatch()
        elif np.all(facecolors[:, 3] == facecolors[0, 3]):
            filled = facecolors[0, 3] != 0.0
        else:
            can_do_optimization = False
        if not len(edgecolors):
            stroked = False
        elif np.all(np.asarray(linewidths) == 0.0):
            stroked = False
        elif np.all(edgecolors[:, 3] == edgecolors[0, 3]):
            stroked = edgecolors[0, 3] != 0.0
        else:
            can_do_optimization = False
        len_path = len(paths[0].vertices) if len(paths) > 0 else 0
        uses_per_path = self._iter_collection_uses_per_path(paths, all_transforms, offsets, facecolors, edgecolors)
        should_do_optimization = len_path + uses_per_path + 5 < len_path * uses_per_path
        if not can_do_optimization or not should_do_optimization:
            return RendererBase.draw_path_collection(self, gc, master_transform, paths, all_transforms, offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position)
        padding = np.max(linewidths)
        path_codes = []
        for i, (path, transform) in enumerate(self._iter_collection_raw_paths(master_transform, paths, all_transforms)):
            name = self.file.pathCollectionObject(gc, path, transform, padding, filled, stroked)
            path_codes.append(name)
        output = self.file.output
        output(*self.gc.push())
        lastx, lasty = (0, 0)
        for xo, yo, path_id, gc0, rgbFace in self._iter_collection(gc, path_codes, offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position):
            self.check_gc(gc0, rgbFace)
            dx, dy = (xo - lastx, yo - lasty)
            output(1, 0, 0, 1, dx, dy, Op.concat_matrix, path_id, Op.use_xobject)
            lastx, lasty = (xo, yo)
        output(*self.gc.pop())

    def draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
        len_marker_path = len(marker_path)
        uses = len(path)
        if len_marker_path * uses < len_marker_path + uses + 5:
            RendererBase.draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace)
            return
        self.check_gc(gc, rgbFace)
        fill = gc.fill(rgbFace)
        stroke = gc.stroke()
        output = self.file.output
        marker = self.file.markerObject(marker_path, marker_trans, fill, stroke, self.gc._linewidth, gc.get_joinstyle(), gc.get_capstyle())
        output(Op.gsave)
        lastx, lasty = (0, 0)
        for vertices, code in path.iter_segments(trans, clip=(0, 0, self.file.width * 72, self.file.height * 72), simplify=False):
            if len(vertices):
                x, y = vertices[-2:]
                if not (0 <= x <= self.file.width * 72 and 0 <= y <= self.file.height * 72):
                    continue
                dx, dy = (x - lastx, y - lasty)
                output(1, 0, 0, 1, dx, dy, Op.concat_matrix, marker, Op.use_xobject)
                lastx, lasty = (x, y)
        output(Op.grestore)

    def draw_gouraud_triangle(self, gc, points, colors, trans):
        self.draw_gouraud_triangles(gc, points.reshape((1, 3, 2)), colors.reshape((1, 3, 4)), trans)

    def draw_gouraud_triangles(self, gc, points, colors, trans):
        assert len(points) == len(colors)
        if len(points) == 0:
            return
        assert points.ndim == 3
        assert points.shape[1] == 3
        assert points.shape[2] == 2
        assert colors.ndim == 3
        assert colors.shape[1] == 3
        assert colors.shape[2] in (1, 4)
        shape = points.shape
        points = points.reshape((shape[0] * shape[1], 2))
        tpoints = trans.transform(points)
        tpoints = tpoints.reshape(shape)
        name, _ = self.file.addGouraudTriangles(tpoints, colors)
        output = self.file.output
        if colors.shape[2] == 1:
            gc.set_alpha(1.0)
            self.check_gc(gc)
            output(name, Op.shading)
            return
        alpha = colors[0, 0, 3]
        if np.allclose(alpha, colors[:, :, 3]):
            gc.set_alpha(alpha)
            self.check_gc(gc)
            output(name, Op.shading)
        else:
            alpha = colors[:, :, 3][:, :, None]
            _, smask_ob = self.file.addGouraudTriangles(tpoints, alpha)
            gstate = self.file._soft_mask_state(smask_ob)
            output(Op.gsave, gstate, Op.setgstate, name, Op.shading, Op.grestore)

    def _setup_textpos(self, x, y, angle, oldx=0, oldy=0, oldangle=0):
        if angle == oldangle == 0:
            self.file.output(x - oldx, y - oldy, Op.textpos)
        else:
            angle = math.radians(angle)
            self.file.output(math.cos(angle), math.sin(angle), -math.sin(angle), math.cos(angle), x, y, Op.textmatrix)
            self.file.output(0, 0, Op.textpos)

    def draw_mathtext(self, gc, x, y, s, prop, angle):
        width, height, descent, glyphs, rects = self._text2path.mathtext_parser.parse(s, 72, prop)
        if gc.get_url() is not None:
            self.file._annotations[-1][1].append(_get_link_annotation(gc, x, y, width, height, angle))
        fonttype = mpl.rcParams['pdf.fonttype']
        a = math.radians(angle)
        self.file.output(Op.gsave)
        self.file.output(math.cos(a), math.sin(a), -math.sin(a), math.cos(a), x, y, Op.concat_matrix)
        self.check_gc(gc, gc._rgb)
        prev_font = (None, None)
        oldx, oldy = (0, 0)
        unsupported_chars = []
        self.file.output(Op.begin_text)
        for font, fontsize, num, ox, oy in glyphs:
            self.file._character_tracker.track_glyph(font, num)
            fontname = font.fname
            if not _font_supports_glyph(fonttype, num):
                unsupported_chars.append((font, fontsize, ox, oy, num))
            else:
                self._setup_textpos(ox, oy, 0, oldx, oldy)
                oldx, oldy = (ox, oy)
                if (fontname, fontsize) != prev_font:
                    self.file.output(self.file.fontName(fontname), fontsize, Op.selectfont)
                    prev_font = (fontname, fontsize)
                self.file.output(self.encode_string(chr(num), fonttype), Op.show)
        self.file.output(Op.end_text)
        for font, fontsize, ox, oy, num in unsupported_chars:
            self._draw_xobject_glyph(font, fontsize, font.get_char_index(num), ox, oy)
        for ox, oy, width, height in rects:
            self.file.output(Op.gsave, ox, oy, width, height, Op.rectangle, Op.fill, Op.grestore)
        self.file.output(Op.grestore)

    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext=None):
        texmanager = self.get_texmanager()
        fontsize = prop.get_size_in_points()
        dvifile = texmanager.make_dvi(s, fontsize)
        with dviread.Dvi(dvifile, 72) as dvi:
            page, = dvi
        if gc.get_url() is not None:
            self.file._annotations[-1][1].append(_get_link_annotation(gc, x, y, page.width, page.height, angle))
        oldfont, seq = (None, [])
        for x1, y1, dvifont, glyph, width in page.text:
            if dvifont != oldfont:
                pdfname = self.file.dviFontName(dvifont)
                seq += [['font', pdfname, dvifont.size]]
                oldfont = dvifont
            seq += [['text', x1, y1, [bytes([glyph])], x1 + width]]
        i, curx, fontsize = (0, 0, None)
        while i < len(seq) - 1:
            elt, nxt = seq[i:i + 2]
            if elt[0] == 'font':
                fontsize = elt[2]
            elif elt[0] == nxt[0] == 'text' and elt[2] == nxt[2]:
                offset = elt[4] - nxt[1]
                if abs(offset) < 0.1:
                    elt[3][-1] += nxt[3][0]
                    elt[4] += nxt[4] - nxt[1]
                else:
                    elt[3] += [offset * 1000.0 / fontsize, nxt[3][0]]
                    elt[4] = nxt[4]
                del seq[i + 1]
                continue
            i += 1
        mytrans = Affine2D().rotate_deg(angle).translate(x, y)
        self.check_gc(gc, gc._rgb)
        self.file.output(Op.begin_text)
        curx, cury, oldx, oldy = (0, 0, 0, 0)
        for elt in seq:
            if elt[0] == 'font':
                self.file.output(elt[1], elt[2], Op.selectfont)
            elif elt[0] == 'text':
                curx, cury = mytrans.transform((elt[1], elt[2]))
                self._setup_textpos(curx, cury, angle, oldx, oldy)
                oldx, oldy = (curx, cury)
                if len(elt[3]) == 1:
                    self.file.output(elt[3][0], Op.show)
                else:
                    self.file.output(elt[3], Op.showkern)
            else:
                assert False
        self.file.output(Op.end_text)
        boxgc = self.new_gc()
        boxgc.copy_properties(gc)
        boxgc.set_linewidth(0)
        pathops = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        for x1, y1, h, w in page.boxes:
            path = Path([[x1, y1], [x1 + w, y1], [x1 + w, y1 + h], [x1, y1 + h], [0, 0]], pathops)
            self.draw_path(boxgc, path, mytrans, gc._rgb)

    def encode_string(self, s, fonttype):
        if fonttype in (1, 3):
            return s.encode('cp1252', 'replace')
        return s.encode('utf-16be', 'replace')

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        self.check_gc(gc, gc._rgb)
        if ismath:
            return self.draw_mathtext(gc, x, y, s, prop, angle)
        fontsize = prop.get_size_in_points()
        if mpl.rcParams['pdf.use14corefonts']:
            font = self._get_font_afm(prop)
            fonttype = 1
        else:
            font = self._get_font_ttf(prop)
            self.file._character_tracker.track(font, s)
            fonttype = mpl.rcParams['pdf.fonttype']
        if gc.get_url() is not None:
            font.set_text(s)
            width, height = font.get_width_height()
            self.file._annotations[-1][1].append(_get_link_annotation(gc, x, y, width / 64, height / 64, angle))
        if fonttype not in [3, 42]:
            self.file.output(Op.begin_text, self.file.fontName(prop), fontsize, Op.selectfont)
            self._setup_textpos(x, y, angle)
            self.file.output(self.encode_string(s, fonttype), Op.show, Op.end_text)
        else:
            singlebyte_chunks = []
            multibyte_glyphs = []
            prev_was_multibyte = True
            prev_font = font
            for item in _text_helpers.layout(s, font, kern_mode=KERNING_UNFITTED):
                if _font_supports_glyph(fonttype, ord(item.char)):
                    if prev_was_multibyte or item.ft_object != prev_font:
                        singlebyte_chunks.append((item.ft_object, item.x, []))
                        prev_font = item.ft_object
                    if item.prev_kern:
                        singlebyte_chunks[-1][2].append(item.prev_kern)
                    singlebyte_chunks[-1][2].append(item.char)
                    prev_was_multibyte = False
                else:
                    multibyte_glyphs.append((item.ft_object, item.x, item.glyph_idx))
                    prev_was_multibyte = True
            self.file.output(Op.gsave)
            a = math.radians(angle)
            self.file.output(math.cos(a), math.sin(a), -math.sin(a), math.cos(a), x, y, Op.concat_matrix)
            self.file.output(Op.begin_text)
            prev_start_x = 0
            for ft_object, start_x, kerns_or_chars in singlebyte_chunks:
                ft_name = self.file.fontName(ft_object.fname)
                self.file.output(ft_name, fontsize, Op.selectfont)
                self._setup_textpos(start_x, 0, 0, prev_start_x, 0, 0)
                self.file.output([-1000 * next(group) / fontsize if tp == float else self.encode_string(''.join(group), fonttype) for tp, group in itertools.groupby(kerns_or_chars, type)], Op.showkern)
                prev_start_x = start_x
            self.file.output(Op.end_text)
            for ft_object, start_x, glyph_idx in multibyte_glyphs:
                self._draw_xobject_glyph(ft_object, fontsize, glyph_idx, start_x, 0)
            self.file.output(Op.grestore)

    def _draw_xobject_glyph(self, font, fontsize, glyph_idx, x, y):
        """Draw a multibyte character from a Type 3 font as an XObject."""
        glyph_name = font.get_glyph_name(glyph_idx)
        name = self.file._get_xobject_glyph_name(font.fname, glyph_name)
        self.file.output(Op.gsave, 0.001 * fontsize, 0, 0, 0.001 * fontsize, x, y, Op.concat_matrix, Name(name), Op.use_xobject, Op.grestore)

    def new_gc(self):
        return GraphicsContextPdf(self.file)