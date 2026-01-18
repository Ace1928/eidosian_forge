import codecs
import datetime
from enum import Enum
import functools
from io import StringIO
import itertools
import logging
import os
import pathlib
import shutil
from tempfile import TemporaryDirectory
import time
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _path, _text_helpers
from matplotlib._afm import AFM
from matplotlib.backend_bases import (
from matplotlib.cbook import is_writable_file_like, file_requires_unicode
from matplotlib.font_manager import get_font
from matplotlib.ft2font import LOAD_NO_SCALE, FT2Font
from matplotlib._ttconv import convert_ttf_to_ps
from matplotlib._mathtext_data import uni2type1
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib.backends.backend_mixed import MixedModeRenderer
from . import _backend_pdf_ps
class RendererPS(_backend_pdf_ps.RendererPDFPSBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles.
    """
    _afm_font_dir = cbook._get_data_path('fonts/afm')
    _use_afm_rc_name = 'ps.useafm'

    def __init__(self, width, height, pswriter, imagedpi=72):
        super().__init__(width, height)
        self._pswriter = pswriter
        if mpl.rcParams['text.usetex']:
            self.textcnt = 0
            self.psfrag = []
        self.imagedpi = imagedpi
        self.color = None
        self.linewidth = None
        self.linejoin = None
        self.linecap = None
        self.linedash = None
        self.fontname = None
        self.fontsize = None
        self._hatches = {}
        self.image_magnification = imagedpi / 72
        self._clip_paths = {}
        self._path_collection_id = 0
        self._character_tracker = _backend_pdf_ps.CharacterTracker()
        self._logwarn_once = functools.cache(_log.warning)

    def _is_transparent(self, rgb_or_rgba):
        if rgb_or_rgba is None:
            return True
        elif len(rgb_or_rgba) == 4:
            if rgb_or_rgba[3] == 0:
                return True
            if rgb_or_rgba[3] != 1:
                self._logwarn_once('The PostScript backend does not support transparency; partially transparent artists will be rendered opaque.')
            return False
        else:
            return False

    def set_color(self, r, g, b, store=True):
        if (r, g, b) != self.color:
            self._pswriter.write(f'{_nums_to_str(r)} setgray\n' if r == g == b else f'{_nums_to_str(r, g, b)} setrgbcolor\n')
            if store:
                self.color = (r, g, b)

    def set_linewidth(self, linewidth, store=True):
        linewidth = float(linewidth)
        if linewidth != self.linewidth:
            self._pswriter.write(f'{_nums_to_str(linewidth)} setlinewidth\n')
            if store:
                self.linewidth = linewidth

    @staticmethod
    def _linejoin_cmd(linejoin):
        linejoin = {'miter': 0, 'round': 1, 'bevel': 2, 0: 0, 1: 1, 2: 2}[linejoin]
        return f'{linejoin:d} setlinejoin\n'

    def set_linejoin(self, linejoin, store=True):
        if linejoin != self.linejoin:
            self._pswriter.write(self._linejoin_cmd(linejoin))
            if store:
                self.linejoin = linejoin

    @staticmethod
    def _linecap_cmd(linecap):
        linecap = {'butt': 0, 'round': 1, 'projecting': 2, 0: 0, 1: 1, 2: 2}[linecap]
        return f'{linecap:d} setlinecap\n'

    def set_linecap(self, linecap, store=True):
        if linecap != self.linecap:
            self._pswriter.write(self._linecap_cmd(linecap))
            if store:
                self.linecap = linecap

    def set_linedash(self, offset, seq, store=True):
        if self.linedash is not None:
            oldo, oldseq = self.linedash
            if np.array_equal(seq, oldseq) and oldo == offset:
                return
        self._pswriter.write(f'[{_nums_to_str(*seq)}] {_nums_to_str(offset)} setdash\n' if seq is not None and len(seq) else '[] 0 setdash\n')
        if store:
            self.linedash = (offset, seq)

    def set_font(self, fontname, fontsize, store=True):
        if (fontname, fontsize) != (self.fontname, self.fontsize):
            self._pswriter.write(f'/{fontname} {fontsize:1.3f} selectfont\n')
            if store:
                self.fontname = fontname
                self.fontsize = fontsize

    def create_hatch(self, hatch):
        sidelen = 72
        if hatch in self._hatches:
            return self._hatches[hatch]
        name = 'H%d' % len(self._hatches)
        linewidth = mpl.rcParams['hatch.linewidth']
        pageheight = self.height * 72
        self._pswriter.write(f'  << /PatternType 1\n     /PaintType 2\n     /TilingType 2\n     /BBox[0 0 {sidelen:d} {sidelen:d}]\n     /XStep {sidelen:d}\n     /YStep {sidelen:d}\n\n     /PaintProc {{\n        pop\n        {linewidth:g} setlinewidth\n{self._convert_path(Path.hatch(hatch), Affine2D().scale(sidelen), simplify=False)}\n        gsave\n        fill\n        grestore\n        stroke\n     }} bind\n   >>\n   matrix\n   0 {pageheight:g} translate\n   makepattern\n   /{name} exch def\n')
        self._hatches[hatch] = name
        return name

    def get_image_magnification(self):
        """
        Get the factor by which to magnify images passed to draw_image.
        Allows a backend to have images at a different resolution to other
        artists.
        """
        return self.image_magnification

    def _convert_path(self, path, transform, clip=False, simplify=None):
        if clip:
            clip = (0.0, 0.0, self.width * 72.0, self.height * 72.0)
        else:
            clip = None
        return _path.convert_to_string(path, transform, clip, simplify, None, 6, [b'm', b'l', b'', b'c', b'cl'], True).decode('ascii')

    def _get_clip_cmd(self, gc):
        clip = []
        rect = gc.get_clip_rectangle()
        if rect is not None:
            clip.append(f'{_nums_to_str(*rect.p0, *rect.size)} rectclip\n')
        path, trf = gc.get_clip_path()
        if path is not None:
            key = (path, id(trf))
            custom_clip_cmd = self._clip_paths.get(key)
            if custom_clip_cmd is None:
                custom_clip_cmd = 'c%d' % len(self._clip_paths)
                self._pswriter.write(f'/{custom_clip_cmd} {{\n{self._convert_path(path, trf, simplify=False)}\nclip\nnewpath\n}} bind def\n')
                self._clip_paths[key] = custom_clip_cmd
            clip.append(f'{custom_clip_cmd}\n')
        return ''.join(clip)

    @_log_if_debug_on
    def draw_image(self, gc, x, y, im, transform=None):
        h, w = im.shape[:2]
        imagecmd = 'false 3 colorimage'
        data = im[::-1, :, :3]
        hexdata = data.tobytes().hex('\n', -64)
        if transform is None:
            matrix = '1 0 0 1 0 0'
            xscale = w / self.image_magnification
            yscale = h / self.image_magnification
        else:
            matrix = ' '.join(map(str, transform.frozen().to_values()))
            xscale = 1.0
            yscale = 1.0
        self._pswriter.write(f'gsave\n{self._get_clip_cmd(gc)}\n{x:g} {y:g} translate\n[{matrix}] concat\n{xscale:g} {yscale:g} scale\n/DataString {w:d} string def\n{w:d} {h:d} 8 [ {w:d} 0 0 -{h:d} 0 {h:d} ]\n{{\ncurrentfile DataString readhexstring pop\n}} bind {imagecmd}\n{hexdata}\ngrestore\n')

    @_log_if_debug_on
    def draw_path(self, gc, path, transform, rgbFace=None):
        clip = rgbFace is None and gc.get_hatch_path() is None
        simplify = path.should_simplify and clip
        ps = self._convert_path(path, transform, clip=clip, simplify=simplify)
        self._draw_ps(ps, gc, rgbFace)

    @_log_if_debug_on
    def draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
        ps_color = None if self._is_transparent(rgbFace) else f'{_nums_to_str(rgbFace[0])} setgray' if rgbFace[0] == rgbFace[1] == rgbFace[2] else f'{_nums_to_str(*rgbFace[:3])} setrgbcolor'
        ps_cmd = ['/o {', 'gsave', 'newpath', 'translate']
        lw = gc.get_linewidth()
        alpha = gc.get_alpha() if gc.get_forced_alpha() or len(gc.get_rgb()) == 3 else gc.get_rgb()[3]
        stroke = lw > 0 and alpha > 0
        if stroke:
            ps_cmd.append('%.1f setlinewidth' % lw)
            ps_cmd.append(self._linejoin_cmd(gc.get_joinstyle()))
            ps_cmd.append(self._linecap_cmd(gc.get_capstyle()))
        ps_cmd.append(self._convert_path(marker_path, marker_trans, simplify=False))
        if rgbFace:
            if stroke:
                ps_cmd.append('gsave')
            if ps_color:
                ps_cmd.extend([ps_color, 'fill'])
            if stroke:
                ps_cmd.append('grestore')
        if stroke:
            ps_cmd.append('stroke')
        ps_cmd.extend(['grestore', '} bind def'])
        for vertices, code in path.iter_segments(trans, clip=(0, 0, self.width * 72, self.height * 72), simplify=False):
            if len(vertices):
                x, y = vertices[-2:]
                ps_cmd.append(f'{x:g} {y:g} o')
        ps = '\n'.join(ps_cmd)
        self._draw_ps(ps, gc, rgbFace, fill=False, stroke=False)

    @_log_if_debug_on
    def draw_path_collection(self, gc, master_transform, paths, all_transforms, offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position):
        len_path = len(paths[0].vertices) if len(paths) > 0 else 0
        uses_per_path = self._iter_collection_uses_per_path(paths, all_transforms, offsets, facecolors, edgecolors)
        should_do_optimization = len_path + 3 * uses_per_path + 3 < (len_path + 2) * uses_per_path
        if not should_do_optimization:
            return RendererBase.draw_path_collection(self, gc, master_transform, paths, all_transforms, offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position)
        path_codes = []
        for i, (path, transform) in enumerate(self._iter_collection_raw_paths(master_transform, paths, all_transforms)):
            name = 'p%d_%d' % (self._path_collection_id, i)
            path_bytes = self._convert_path(path, transform, simplify=False)
            self._pswriter.write(f'/{name} {{\nnewpath\ntranslate\n{path_bytes}\n}} bind def\n')
            path_codes.append(name)
        for xo, yo, path_id, gc0, rgbFace in self._iter_collection(gc, path_codes, offsets, offset_trans, facecolors, edgecolors, linewidths, linestyles, antialiaseds, urls, offset_position):
            ps = f'{xo:g} {yo:g} {path_id}'
            self._draw_ps(ps, gc0, rgbFace)
        self._path_collection_id += 1

    @_log_if_debug_on
    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext=None):
        if self._is_transparent(gc.get_rgb()):
            return
        if not hasattr(self, 'psfrag'):
            self._logwarn_once("The PS backend determines usetex status solely based on rcParams['text.usetex'] and does not support having usetex=True only for some elements; this element will thus be rendered as if usetex=False.")
            self.draw_text(gc, x, y, s, prop, angle, False, mtext)
            return
        w, h, bl = self.get_text_width_height_descent(s, prop, ismath='TeX')
        fontsize = prop.get_size_in_points()
        thetext = 'psmarker%d' % self.textcnt
        color = _nums_to_str(*gc.get_rgb()[:3], sep=',')
        fontcmd = {'sans-serif': '{\\sffamily %s}', 'monospace': '{\\ttfamily %s}'}.get(mpl.rcParams['font.family'][0], '{\\rmfamily %s}')
        s = fontcmd % s
        tex = '\\color[rgb]{%s} %s' % (color, s)
        rangle = np.radians(angle + 90)
        pos = _nums_to_str(x - bl * np.cos(rangle), y - bl * np.sin(rangle))
        self.psfrag.append('\\psfrag{%s}[bl][bl][1][%f]{\\fontsize{%f}{%f}%s}' % (thetext, angle, fontsize, fontsize * 1.25, tex))
        self._pswriter.write(f'gsave\n{pos} moveto\n({thetext})\nshow\ngrestore\n')
        self.textcnt += 1

    @_log_if_debug_on
    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        if self._is_transparent(gc.get_rgb()):
            return
        if ismath == 'TeX':
            return self.draw_tex(gc, x, y, s, prop, angle)
        if ismath:
            return self.draw_mathtext(gc, x, y, s, prop, angle)
        stream = []
        if mpl.rcParams['ps.useafm']:
            font = self._get_font_afm(prop)
            ps_name = font.postscript_name.encode('ascii', 'replace').decode('ascii')
            scale = 0.001 * prop.get_size_in_points()
            thisx = 0
            last_name = None
            for c in s:
                name = uni2type1.get(ord(c), f'uni{ord(c):04X}')
                try:
                    width = font.get_width_from_char_name(name)
                except KeyError:
                    name = 'question'
                    width = font.get_width_char('?')
                kern = font.get_kern_dist_from_name(last_name, name)
                last_name = name
                thisx += kern * scale
                stream.append((ps_name, thisx, name))
                thisx += width * scale
        else:
            font = self._get_font_ttf(prop)
            self._character_tracker.track(font, s)
            for item in _text_helpers.layout(s, font):
                ps_name = item.ft_object.postscript_name.encode('ascii', 'replace').decode('ascii')
                glyph_name = item.ft_object.get_glyph_name(item.glyph_idx)
                stream.append((ps_name, item.x, glyph_name))
        self.set_color(*gc.get_rgb())
        for ps_name, group in itertools.groupby(stream, lambda entry: entry[0]):
            self.set_font(ps_name, prop.get_size_in_points(), False)
            thetext = '\n'.join((f'{x:g} 0 m /{name:s} glyphshow' for _, x, name in group))
            self._pswriter.write(f'gsave\n{self._get_clip_cmd(gc)}\n{x:g} {y:g} translate\n{angle:g} rotate\n{thetext}\ngrestore\n')

    @_log_if_debug_on
    def draw_mathtext(self, gc, x, y, s, prop, angle):
        """Draw the math text using matplotlib.mathtext."""
        width, height, descent, glyphs, rects = self._text2path.mathtext_parser.parse(s, 72, prop)
        self.set_color(*gc.get_rgb())
        self._pswriter.write(f'gsave\n{x:g} {y:g} translate\n{angle:g} rotate\n')
        lastfont = None
        for font, fontsize, num, ox, oy in glyphs:
            self._character_tracker.track_glyph(font, num)
            if (font.postscript_name, fontsize) != lastfont:
                lastfont = (font.postscript_name, fontsize)
                self._pswriter.write(f'/{font.postscript_name} {fontsize} selectfont\n')
            glyph_name = font.get_name_char(chr(num)) if isinstance(font, AFM) else font.get_glyph_name(font.get_char_index(num))
            self._pswriter.write(f'{ox:g} {oy:g} moveto\n/{glyph_name} glyphshow\n')
        for ox, oy, w, h in rects:
            self._pswriter.write(f'{ox} {oy} {w} {h} rectfill\n')
        self._pswriter.write('grestore\n')

    @_log_if_debug_on
    def draw_gouraud_triangle(self, gc, points, colors, trans):
        self.draw_gouraud_triangles(gc, points.reshape((1, 3, 2)), colors.reshape((1, 3, 4)), trans)

    @_log_if_debug_on
    def draw_gouraud_triangles(self, gc, points, colors, trans):
        assert len(points) == len(colors)
        if len(points) == 0:
            return
        assert points.ndim == 3
        assert points.shape[1] == 3
        assert points.shape[2] == 2
        assert colors.ndim == 3
        assert colors.shape[1] == 3
        assert colors.shape[2] == 4
        shape = points.shape
        flat_points = points.reshape((shape[0] * shape[1], 2))
        flat_points = trans.transform(flat_points)
        flat_colors = colors.reshape((shape[0] * shape[1], 4))
        points_min = np.min(flat_points, axis=0) - (1 << 12)
        points_max = np.max(flat_points, axis=0) + (1 << 12)
        factor = np.ceil((2 ** 32 - 1) / (points_max - points_min))
        xmin, ymin = points_min
        xmax, ymax = points_max
        data = np.empty(shape[0] * shape[1], dtype=[('flags', 'u1'), ('points', '2>u4'), ('colors', '3u1')])
        data['flags'] = 0
        data['points'] = (flat_points - points_min) * factor
        data['colors'] = flat_colors[:, :3] * 255.0
        hexdata = data.tobytes().hex('\n', -64)
        self._pswriter.write(f'gsave\n<< /ShadingType 4\n   /ColorSpace [/DeviceRGB]\n   /BitsPerCoordinate 32\n   /BitsPerComponent 8\n   /BitsPerFlag 8\n   /AntiAlias true\n   /Decode [ {xmin:g} {xmax:g} {ymin:g} {ymax:g} 0 1 0 1 0 1 ]\n   /DataSource <\n{hexdata}\n>\n>>\nshfill\ngrestore\n')

    def _draw_ps(self, ps, gc, rgbFace, *, fill=True, stroke=True):
        """
        Emit the PostScript snippet *ps* with all the attributes from *gc*
        applied.  *ps* must consist of PostScript commands to construct a path.

        The *fill* and/or *stroke* kwargs can be set to False if the *ps*
        string already includes filling and/or stroking, in which case
        `_draw_ps` is just supplying properties and clipping.
        """
        write = self._pswriter.write
        mightstroke = gc.get_linewidth() > 0 and (not self._is_transparent(gc.get_rgb()))
        if not mightstroke:
            stroke = False
        if self._is_transparent(rgbFace):
            fill = False
        hatch = gc.get_hatch()
        if mightstroke:
            self.set_linewidth(gc.get_linewidth())
            self.set_linejoin(gc.get_joinstyle())
            self.set_linecap(gc.get_capstyle())
            self.set_linedash(*gc.get_dashes())
        if mightstroke or hatch:
            self.set_color(*gc.get_rgb()[:3])
        write('gsave\n')
        write(self._get_clip_cmd(gc))
        write(ps.strip())
        write('\n')
        if fill:
            if stroke or hatch:
                write('gsave\n')
            self.set_color(*rgbFace[:3], store=False)
            write('fill\n')
            if stroke or hatch:
                write('grestore\n')
        if hatch:
            hatch_name = self.create_hatch(hatch)
            write('gsave\n')
            write(_nums_to_str(*gc.get_hatch_color()[:3]))
            write(f' {hatch_name} setpattern fill grestore\n')
        if stroke:
            write('stroke\n')
        write('grestore\n')