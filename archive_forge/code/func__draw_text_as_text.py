import base64
import codecs
import datetime
import gzip
import hashlib
from io import BytesIO
import itertools
import logging
import os
import re
import uuid
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import cbook, font_manager as fm
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.colors import rgb2hex
from matplotlib.dates import UTC
from matplotlib.path import Path
from matplotlib import _path
from matplotlib.transforms import Affine2D, Affine2DBase
def _draw_text_as_text(self, gc, x, y, s, prop, angle, ismath, mtext=None):
    writer = self.writer
    color = rgb2hex(gc.get_rgb())
    style = {}
    if color != '#000000':
        style['fill'] = color
    alpha = gc.get_alpha() if gc.get_forced_alpha() else gc.get_rgb()[3]
    if alpha != 1:
        style['opacity'] = _short_float_fmt(alpha)
    if not ismath:
        attrib = {}
        font_parts = []
        if prop.get_style() != 'normal':
            font_parts.append(prop.get_style())
        if prop.get_variant() != 'normal':
            font_parts.append(prop.get_variant())
        weight = fm.weight_dict[prop.get_weight()]
        if weight != 400:
            font_parts.append(f'{weight}')

        def _normalize_sans(name):
            return 'sans-serif' if name in ['sans', 'sans serif'] else name

        def _expand_family_entry(fn):
            fn = _normalize_sans(fn)
            if fn in fm.font_family_aliases:
                for name in fm.FontManager._expand_aliases(fn):
                    yield _normalize_sans(name)
            yield fn

        def _get_all_quoted_names(prop):
            return [name if name in fm.font_family_aliases else repr(name) for entry in prop.get_family() for name in _expand_family_entry(entry)]
        font_parts.extend([f'{_short_float_fmt(prop.get_size())}px', ', '.join(dict.fromkeys(_get_all_quoted_names(prop)))])
        style['font'] = ' '.join(font_parts)
        if prop.get_stretch() != 'normal':
            style['font-stretch'] = prop.get_stretch()
        attrib['style'] = _generate_css(style)
        if mtext and (angle == 0 or mtext.get_rotation_mode() == 'anchor'):
            transform = mtext.get_transform()
            ax, ay = transform.transform(mtext.get_unitless_position())
            ay = self.height - ay
            angle_rad = np.deg2rad(angle)
            dir_vert = np.array([np.sin(angle_rad), np.cos(angle_rad)])
            v_offset = np.dot(dir_vert, [x - ax, y - ay])
            ax = ax + v_offset * dir_vert[0]
            ay = ay + v_offset * dir_vert[1]
            ha_mpl_to_svg = {'left': 'start', 'right': 'end', 'center': 'middle'}
            style['text-anchor'] = ha_mpl_to_svg[mtext.get_ha()]
            attrib['x'] = _short_float_fmt(ax)
            attrib['y'] = _short_float_fmt(ay)
            attrib['style'] = _generate_css(style)
            attrib['transform'] = _generate_transform([('rotate', (-angle, ax, ay))])
        else:
            attrib['transform'] = _generate_transform([('translate', (x, y)), ('rotate', (-angle,))])
        writer.element('text', s, attrib=attrib)
    else:
        writer.comment(s)
        width, height, descent, glyphs, rects = self._text2path.mathtext_parser.parse(s, 72, prop)
        writer.start('g', style=_generate_css(style), transform=_generate_transform([('translate', (x, y)), ('rotate', (-angle,))]))
        writer.start('text')
        spans = {}
        for font, fontsize, thetext, new_x, new_y in glyphs:
            entry = fm.ttfFontProperty(font)
            font_parts = []
            if entry.style != 'normal':
                font_parts.append(entry.style)
            if entry.variant != 'normal':
                font_parts.append(entry.variant)
            if entry.weight != 400:
                font_parts.append(f'{entry.weight}')
            font_parts.extend([f'{_short_float_fmt(fontsize)}px', f'{entry.name!r}'])
            style = {'font': ' '.join(font_parts)}
            if entry.stretch != 'normal':
                style['font-stretch'] = entry.stretch
            style = _generate_css(style)
            if thetext == 32:
                thetext = 160
            spans.setdefault(style, []).append((new_x, -new_y, thetext))
        for style, chars in spans.items():
            chars.sort()
            if len({y for x, y, t in chars}) == 1:
                ys = str(chars[0][1])
            else:
                ys = ' '.join((str(c[1]) for c in chars))
            attrib = {'style': style, 'x': ' '.join((_short_float_fmt(c[0]) for c in chars)), 'y': ys}
            writer.element('tspan', ''.join((chr(c[2]) for c in chars)), attrib=attrib)
        writer.end('text')
        for x, y, width, height in rects:
            writer.element('rect', x=_short_float_fmt(x), y=_short_float_fmt(-y - 1), width=_short_float_fmt(width), height=_short_float_fmt(height))
        writer.end('g')