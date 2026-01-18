import functools
import gzip
import math
import numpy as np
from .. import _api, cbook, font_manager
from matplotlib.backend_bases import (
from matplotlib.font_manager import ttfFontProperty
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
class RendererCairo(RendererBase):

    def __init__(self, dpi):
        self.dpi = dpi
        self.gc = GraphicsContextCairo(renderer=self)
        self.width = None
        self.height = None
        self.text_ctx = cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1))
        super().__init__()

    def set_context(self, ctx):
        surface = ctx.get_target()
        if hasattr(surface, 'get_width') and hasattr(surface, 'get_height'):
            size = (surface.get_width(), surface.get_height())
        elif hasattr(surface, 'get_extents'):
            ext = surface.get_extents()
            size = (ext.width, ext.height)
        else:
            ctx.save()
            ctx.reset_clip()
            rect, *rest = ctx.copy_clip_rectangle_list()
            if rest:
                raise TypeError('Cannot infer surface size')
            _, _, *size = rect
            ctx.restore()
        self.gc.ctx = ctx
        self.width, self.height = size

    def _fill_and_stroke(self, ctx, fill_c, alpha, alpha_overrides):
        if fill_c is not None:
            ctx.save()
            if len(fill_c) == 3 or alpha_overrides:
                ctx.set_source_rgba(fill_c[0], fill_c[1], fill_c[2], alpha)
            else:
                ctx.set_source_rgba(fill_c[0], fill_c[1], fill_c[2], fill_c[3])
            ctx.fill_preserve()
            ctx.restore()
        ctx.stroke()

    def draw_path(self, gc, path, transform, rgbFace=None):
        ctx = gc.ctx
        clip = ctx.clip_extents() if rgbFace is None and gc.get_hatch() is None else None
        transform = transform + Affine2D().scale(1, -1).translate(0, self.height)
        ctx.new_path()
        _append_path(ctx, path, transform, clip)
        self._fill_and_stroke(ctx, rgbFace, gc.get_alpha(), gc.get_forced_alpha())

    def draw_markers(self, gc, marker_path, marker_trans, path, transform, rgbFace=None):
        ctx = gc.ctx
        ctx.new_path()
        _append_path(ctx, marker_path, marker_trans + Affine2D().scale(1, -1))
        marker_path = ctx.copy_path_flat()
        x1, y1, x2, y2 = ctx.fill_extents()
        if x1 == 0 and y1 == 0 and (x2 == 0) and (y2 == 0):
            filled = False
            rgbFace = None
        else:
            filled = True
        transform = transform + Affine2D().scale(1, -1).translate(0, self.height)
        ctx.new_path()
        for i, (vertices, codes) in enumerate(path.iter_segments(transform, simplify=False)):
            if len(vertices):
                x, y = vertices[-2:]
                ctx.save()
                ctx.translate(x, y)
                ctx.append_path(marker_path)
                ctx.restore()
                if filled or i % 1000 == 0:
                    self._fill_and_stroke(ctx, rgbFace, gc.get_alpha(), gc.get_forced_alpha())
        if not filled:
            self._fill_and_stroke(ctx, rgbFace, gc.get_alpha(), gc.get_forced_alpha())

    def draw_image(self, gc, x, y, im):
        im = cbook._unmultiplied_rgba8888_to_premultiplied_argb32(im[::-1])
        surface = cairo.ImageSurface.create_for_data(im.ravel().data, cairo.FORMAT_ARGB32, im.shape[1], im.shape[0], im.shape[1] * 4)
        ctx = gc.ctx
        y = self.height - y - im.shape[0]
        ctx.save()
        ctx.set_source_surface(surface, float(x), float(y))
        ctx.paint()
        ctx.restore()

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        if ismath:
            self._draw_mathtext(gc, x, y, s, prop, angle)
        else:
            ctx = gc.ctx
            ctx.new_path()
            ctx.move_to(x, y)
            ctx.save()
            ctx.select_font_face(*_cairo_font_args_from_font_prop(prop))
            ctx.set_font_size(self.points_to_pixels(prop.get_size_in_points()))
            opts = cairo.FontOptions()
            opts.set_antialias(gc.get_antialiased())
            ctx.set_font_options(opts)
            if angle:
                ctx.rotate(np.deg2rad(-angle))
            ctx.show_text(s)
            ctx.restore()

    def _draw_mathtext(self, gc, x, y, s, prop, angle):
        ctx = gc.ctx
        width, height, descent, glyphs, rects = self._text2path.mathtext_parser.parse(s, self.dpi, prop)
        ctx.save()
        ctx.translate(x, y)
        if angle:
            ctx.rotate(np.deg2rad(-angle))
        for font, fontsize, idx, ox, oy in glyphs:
            ctx.new_path()
            ctx.move_to(ox, -oy)
            ctx.select_font_face(*_cairo_font_args_from_font_prop(ttfFontProperty(font)))
            ctx.set_font_size(self.points_to_pixels(fontsize))
            ctx.show_text(chr(idx))
        for ox, oy, w, h in rects:
            ctx.new_path()
            ctx.rectangle(ox, -oy, w, -h)
            ctx.set_source_rgb(0, 0, 0)
            ctx.fill_preserve()
        ctx.restore()

    def get_canvas_width_height(self):
        return (self.width, self.height)

    def get_text_width_height_descent(self, s, prop, ismath):
        if ismath == 'TeX':
            return super().get_text_width_height_descent(s, prop, ismath)
        if ismath:
            width, height, descent, *_ = self._text2path.mathtext_parser.parse(s, self.dpi, prop)
            return (width, height, descent)
        ctx = self.text_ctx
        ctx.save()
        ctx.select_font_face(*_cairo_font_args_from_font_prop(prop))
        ctx.set_font_size(self.points_to_pixels(prop.get_size_in_points()))
        y_bearing, w, h = ctx.text_extents(s)[1:4]
        ctx.restore()
        return (w, h, h + y_bearing)

    def new_gc(self):
        self.gc.ctx.save()
        self.gc._alpha = 1
        self.gc._forced_alpha = False
        return self.gc

    def points_to_pixels(self, points):
        return points / 72 * self.dpi