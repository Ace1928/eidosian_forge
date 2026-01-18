import codecs
import datetime
import functools
from io import BytesIO
import logging
import math
import os
import pathlib
import shutil
import subprocess
from tempfile import TemporaryDirectory
import weakref
from PIL import Image
import matplotlib as mpl
from matplotlib import _api, cbook, font_manager as fm
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.backends.backend_pdf import (
from matplotlib.path import Path
from matplotlib.figure import Figure
from matplotlib._pylab_helpers import Gcf
class RendererPgf(RendererBase):

    def __init__(self, figure, fh):
        """
        Create a new PGF renderer that translates any drawing instruction
        into text commands to be interpreted in a latex pgfpicture environment.

        Attributes
        ----------
        figure : `~matplotlib.figure.Figure`
            Matplotlib figure to initialize height, width and dpi from.
        fh : file-like
            File handle for the output of the drawing commands.
        """
        super().__init__()
        self.dpi = figure.dpi
        self.fh = fh
        self.figure = figure
        self.image_counter = 0

    def draw_markers(self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
        _writeln(self.fh, '\\begin{pgfscope}')
        f = 1.0 / self.dpi
        self._print_pgf_clip(gc)
        self._print_pgf_path_styles(gc, rgbFace)
        bl, tr = marker_path.get_extents(marker_trans).get_points()
        coords = (bl[0] * f, bl[1] * f, tr[0] * f, tr[1] * f)
        _writeln(self.fh, '\\pgfsys@defobject{currentmarker}{\\pgfqpoint{%fin}{%fin}}{\\pgfqpoint{%fin}{%fin}}{' % coords)
        self._print_pgf_path(None, marker_path, marker_trans)
        self._pgf_path_draw(stroke=gc.get_linewidth() != 0.0, fill=rgbFace is not None)
        _writeln(self.fh, '}')
        maxcoord = 16383 / 72.27 * self.dpi
        clip = (-maxcoord, -maxcoord, maxcoord, maxcoord)
        for point, code in path.iter_segments(trans, simplify=False, clip=clip):
            x, y = (point[0] * f, point[1] * f)
            _writeln(self.fh, '\\begin{pgfscope}')
            _writeln(self.fh, '\\pgfsys@transformshift{%fin}{%fin}' % (x, y))
            _writeln(self.fh, '\\pgfsys@useobject{currentmarker}{}')
            _writeln(self.fh, '\\end{pgfscope}')
        _writeln(self.fh, '\\end{pgfscope}')

    def draw_path(self, gc, path, transform, rgbFace=None):
        _writeln(self.fh, '\\begin{pgfscope}')
        self._print_pgf_clip(gc)
        self._print_pgf_path_styles(gc, rgbFace)
        self._print_pgf_path(gc, path, transform, rgbFace)
        self._pgf_path_draw(stroke=gc.get_linewidth() != 0.0, fill=rgbFace is not None)
        _writeln(self.fh, '\\end{pgfscope}')
        if gc.get_hatch():
            _writeln(self.fh, '\\begin{pgfscope}')
            self._print_pgf_path_styles(gc, rgbFace)
            self._print_pgf_clip(gc)
            self._print_pgf_path(gc, path, transform, rgbFace)
            _writeln(self.fh, '\\pgfusepath{clip}')
            _writeln(self.fh, '\\pgfsys@defobject{currentpattern}{\\pgfqpoint{0in}{0in}}{\\pgfqpoint{1in}{1in}}{')
            _writeln(self.fh, '\\begin{pgfscope}')
            _writeln(self.fh, '\\pgfpathrectangle{\\pgfqpoint{0in}{0in}}{\\pgfqpoint{1in}{1in}}')
            _writeln(self.fh, '\\pgfusepath{clip}')
            scale = mpl.transforms.Affine2D().scale(self.dpi)
            self._print_pgf_path(None, gc.get_hatch_path(), scale)
            self._pgf_path_draw(stroke=True)
            _writeln(self.fh, '\\end{pgfscope}')
            _writeln(self.fh, '}')
            f = 1.0 / self.dpi
            (xmin, ymin), (xmax, ymax) = path.get_extents(transform).get_points()
            xmin, xmax = (f * xmin, f * xmax)
            ymin, ymax = (f * ymin, f * ymax)
            repx, repy = (math.ceil(xmax - xmin), math.ceil(ymax - ymin))
            _writeln(self.fh, '\\pgfsys@transformshift{%fin}{%fin}' % (xmin, ymin))
            for iy in range(repy):
                for ix in range(repx):
                    _writeln(self.fh, '\\pgfsys@useobject{currentpattern}{}')
                    _writeln(self.fh, '\\pgfsys@transformshift{1in}{0in}')
                _writeln(self.fh, '\\pgfsys@transformshift{-%din}{0in}' % repx)
                _writeln(self.fh, '\\pgfsys@transformshift{0in}{1in}')
            _writeln(self.fh, '\\end{pgfscope}')

    def _print_pgf_clip(self, gc):
        f = 1.0 / self.dpi
        bbox = gc.get_clip_rectangle()
        if bbox:
            p1, p2 = bbox.get_points()
            w, h = p2 - p1
            coords = (p1[0] * f, p1[1] * f, w * f, h * f)
            _writeln(self.fh, '\\pgfpathrectangle{\\pgfqpoint{%fin}{%fin}}{\\pgfqpoint{%fin}{%fin}}' % coords)
            _writeln(self.fh, '\\pgfusepath{clip}')
        clippath, clippath_trans = gc.get_clip_path()
        if clippath is not None:
            self._print_pgf_path(gc, clippath, clippath_trans)
            _writeln(self.fh, '\\pgfusepath{clip}')

    def _print_pgf_path_styles(self, gc, rgbFace):
        capstyles = {'butt': '\\pgfsetbuttcap', 'round': '\\pgfsetroundcap', 'projecting': '\\pgfsetrectcap'}
        _writeln(self.fh, capstyles[gc.get_capstyle()])
        joinstyles = {'miter': '\\pgfsetmiterjoin', 'round': '\\pgfsetroundjoin', 'bevel': '\\pgfsetbeveljoin'}
        _writeln(self.fh, joinstyles[gc.get_joinstyle()])
        has_fill = rgbFace is not None
        if gc.get_forced_alpha():
            fillopacity = strokeopacity = gc.get_alpha()
        else:
            strokeopacity = gc.get_rgb()[3]
            fillopacity = rgbFace[3] if has_fill and len(rgbFace) > 3 else 1.0
        if has_fill:
            _writeln(self.fh, '\\definecolor{currentfill}{rgb}{%f,%f,%f}' % tuple(rgbFace[:3]))
            _writeln(self.fh, '\\pgfsetfillcolor{currentfill}')
        if has_fill and fillopacity != 1.0:
            _writeln(self.fh, '\\pgfsetfillopacity{%f}' % fillopacity)
        lw = gc.get_linewidth() * mpl_pt_to_in * latex_in_to_pt
        stroke_rgba = gc.get_rgb()
        _writeln(self.fh, '\\pgfsetlinewidth{%fpt}' % lw)
        _writeln(self.fh, '\\definecolor{currentstroke}{rgb}{%f,%f,%f}' % stroke_rgba[:3])
        _writeln(self.fh, '\\pgfsetstrokecolor{currentstroke}')
        if strokeopacity != 1.0:
            _writeln(self.fh, '\\pgfsetstrokeopacity{%f}' % strokeopacity)
        dash_offset, dash_list = gc.get_dashes()
        if dash_list is None:
            _writeln(self.fh, '\\pgfsetdash{}{0pt}')
        else:
            _writeln(self.fh, '\\pgfsetdash{%s}{%fpt}' % (''.join(('{%fpt}' % dash for dash in dash_list)), dash_offset))

    def _print_pgf_path(self, gc, path, transform, rgbFace=None):
        f = 1.0 / self.dpi
        bbox = gc.get_clip_rectangle() if gc else None
        maxcoord = 16383 / 72.27 * self.dpi
        if bbox and rgbFace is None:
            p1, p2 = bbox.get_points()
            clip = (max(p1[0], -maxcoord), max(p1[1], -maxcoord), min(p2[0], maxcoord), min(p2[1], maxcoord))
        else:
            clip = (-maxcoord, -maxcoord, maxcoord, maxcoord)
        for points, code in path.iter_segments(transform, clip=clip):
            if code == Path.MOVETO:
                x, y = tuple(points)
                _writeln(self.fh, '\\pgfpathmoveto{\\pgfqpoint{%fin}{%fin}}' % (f * x, f * y))
            elif code == Path.CLOSEPOLY:
                _writeln(self.fh, '\\pgfpathclose')
            elif code == Path.LINETO:
                x, y = tuple(points)
                _writeln(self.fh, '\\pgfpathlineto{\\pgfqpoint{%fin}{%fin}}' % (f * x, f * y))
            elif code == Path.CURVE3:
                cx, cy, px, py = tuple(points)
                coords = (cx * f, cy * f, px * f, py * f)
                _writeln(self.fh, '\\pgfpathquadraticcurveto{\\pgfqpoint{%fin}{%fin}}{\\pgfqpoint{%fin}{%fin}}' % coords)
            elif code == Path.CURVE4:
                c1x, c1y, c2x, c2y, px, py = tuple(points)
                coords = (c1x * f, c1y * f, c2x * f, c2y * f, px * f, py * f)
                _writeln(self.fh, '\\pgfpathcurveto{\\pgfqpoint{%fin}{%fin}}{\\pgfqpoint{%fin}{%fin}}{\\pgfqpoint{%fin}{%fin}}' % coords)
        sketch_params = gc.get_sketch_params() if gc else None
        if sketch_params is not None:
            scale, length, randomness = sketch_params
            if scale is not None:
                length *= 0.5
                scale *= 2
                _writeln(self.fh, '\\usepgfmodule{decorations}')
                _writeln(self.fh, '\\usepgflibrary{decorations.pathmorphing}')
                _writeln(self.fh, f'\\pgfkeys{{/pgf/decoration/.cd, segment length = {length * f:f}in, amplitude = {scale * f:f}in}}')
                _writeln(self.fh, f'\\pgfmathsetseed{{{int(randomness)}}}')
                _writeln(self.fh, '\\pgfdecoratecurrentpath{random steps}')

    def _pgf_path_draw(self, stroke=True, fill=False):
        actions = []
        if stroke:
            actions.append('stroke')
        if fill:
            actions.append('fill')
        _writeln(self.fh, '\\pgfusepath{%s}' % ','.join(actions))

    def option_scale_image(self):
        return True

    def option_image_nocomposite(self):
        return not mpl.rcParams['image.composite_image']

    def draw_image(self, gc, x, y, im, transform=None):
        h, w = im.shape[:2]
        if w == 0 or h == 0:
            return
        if not os.path.exists(getattr(self.fh, 'name', '')):
            raise ValueError('streamed pgf-code does not support raster graphics, consider using the pgf-to-pdf option')
        path = pathlib.Path(self.fh.name)
        fname_img = '%s-img%d.png' % (path.stem, self.image_counter)
        Image.fromarray(im[::-1]).save(path.parent / fname_img)
        self.image_counter += 1
        _writeln(self.fh, '\\begin{pgfscope}')
        self._print_pgf_clip(gc)
        f = 1.0 / self.dpi
        if transform is None:
            _writeln(self.fh, '\\pgfsys@transformshift{%fin}{%fin}' % (x * f, y * f))
            w, h = (w * f, h * f)
        else:
            tr1, tr2, tr3, tr4, tr5, tr6 = transform.frozen().to_values()
            _writeln(self.fh, '\\pgfsys@transformcm{%f}{%f}{%f}{%f}{%fin}{%fin}' % (tr1 * f, tr2 * f, tr3 * f, tr4 * f, (tr5 + x) * f, (tr6 + y) * f))
            w = h = 1
        interp = str(transform is None).lower()
        _writeln(self.fh, '\\pgftext[left,bottom]{%s[interpolate=%s,width=%fin,height=%fin]{%s}}' % (_get_image_inclusion_command(), interp, w, h, fname_img))
        _writeln(self.fh, '\\end{pgfscope}')

    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext=None):
        self.draw_text(gc, x, y, s, prop, angle, ismath='TeX', mtext=mtext)

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        s = _escape_and_apply_props(s, prop)
        _writeln(self.fh, '\\begin{pgfscope}')
        self._print_pgf_clip(gc)
        alpha = gc.get_alpha()
        if alpha != 1.0:
            _writeln(self.fh, '\\pgfsetfillopacity{%f}' % alpha)
            _writeln(self.fh, '\\pgfsetstrokeopacity{%f}' % alpha)
        rgb = tuple(gc.get_rgb())[:3]
        _writeln(self.fh, '\\definecolor{textcolor}{rgb}{%f,%f,%f}' % rgb)
        _writeln(self.fh, '\\pgfsetstrokecolor{textcolor}')
        _writeln(self.fh, '\\pgfsetfillcolor{textcolor}')
        s = '\\color{textcolor}' + s
        dpi = self.figure.dpi
        text_args = []
        if mtext and ((angle == 0 or mtext.get_rotation_mode() == 'anchor') and mtext.get_verticalalignment() != 'center_baseline'):
            pos = mtext.get_unitless_position()
            x, y = mtext.get_transform().transform(pos)
            halign = {'left': 'left', 'right': 'right', 'center': ''}
            valign = {'top': 'top', 'bottom': 'bottom', 'baseline': 'base', 'center': ''}
            text_args.extend([f'x={x / dpi:f}in', f'y={y / dpi:f}in', halign[mtext.get_horizontalalignment()], valign[mtext.get_verticalalignment()]])
        else:
            text_args.append(f'x={x / dpi:f}in, y={y / dpi:f}in, left, base')
        if angle != 0:
            text_args.append('rotate=%f' % angle)
        _writeln(self.fh, '\\pgftext[%s]{%s}' % (','.join(text_args), s))
        _writeln(self.fh, '\\end{pgfscope}')

    def get_text_width_height_descent(self, s, prop, ismath):
        w, h, d = LatexManager._get_cached_or_new().get_width_height_descent(s, prop)
        f = mpl_pt_to_in * self.dpi
        return (w * f, h * f, d * f)

    def flipy(self):
        return False

    def get_canvas_width_height(self):
        return (self.figure.get_figwidth() * self.dpi, self.figure.get_figheight() * self.dpi)

    def points_to_pixels(self, points):
        return points * mpl_pt_to_in * self.dpi