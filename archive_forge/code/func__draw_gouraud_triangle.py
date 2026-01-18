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
def _draw_gouraud_triangle(self, gc, points, colors, trans):
    writer = self.writer
    if not self._has_gouraud:
        self._has_gouraud = True
        writer.start('filter', id='colorAdd')
        writer.element('feComposite', attrib={'in': 'SourceGraphic'}, in2='BackgroundImage', operator='arithmetic', k2='1', k3='1')
        writer.end('filter')
        writer.start('filter', id='colorMat')
        writer.element('feColorMatrix', attrib={'type': 'matrix'}, values='1 0 0 0 0 \n0 1 0 0 0 \n0 0 1 0 0' + ' \n1 1 1 1 0 \n0 0 0 0 1 ')
        writer.end('filter')
    avg_color = np.average(colors, axis=0)
    if avg_color[-1] == 0:
        return
    trans_and_flip = self._make_flip_transform(trans)
    tpoints = trans_and_flip.transform(points)
    writer.start('defs')
    for i in range(3):
        x1, y1 = tpoints[i]
        x2, y2 = tpoints[(i + 1) % 3]
        x3, y3 = tpoints[(i + 2) % 3]
        rgba_color = colors[i]
        if x2 == x3:
            xb = x2
            yb = y1
        elif y2 == y3:
            xb = x1
            yb = y2
        else:
            m1 = (y2 - y3) / (x2 - x3)
            b1 = y2 - m1 * x2
            m2 = -(1.0 / m1)
            b2 = y1 - m2 * x1
            xb = (-b1 + b2) / (m1 - m2)
            yb = m2 * xb + b2
        writer.start('linearGradient', id=f'GR{self._n_gradients:x}_{i:d}', gradientUnits='userSpaceOnUse', x1=_short_float_fmt(x1), y1=_short_float_fmt(y1), x2=_short_float_fmt(xb), y2=_short_float_fmt(yb))
        writer.element('stop', offset='1', style=_generate_css({'stop-color': rgb2hex(avg_color), 'stop-opacity': _short_float_fmt(rgba_color[-1])}))
        writer.element('stop', offset='0', style=_generate_css({'stop-color': rgb2hex(rgba_color), 'stop-opacity': '0'}))
        writer.end('linearGradient')
    writer.end('defs')
    dpath = 'M ' + _short_float_fmt(x1) + ',' + _short_float_fmt(y1)
    dpath += ' L ' + _short_float_fmt(x2) + ',' + _short_float_fmt(y2)
    dpath += ' ' + _short_float_fmt(x3) + ',' + _short_float_fmt(y3) + ' Z'
    writer.element('path', attrib={'d': dpath, 'fill': rgb2hex(avg_color), 'fill-opacity': '1', 'shape-rendering': 'crispEdges'})
    writer.start('g', attrib={'stroke': 'none', 'stroke-width': '0', 'shape-rendering': 'crispEdges', 'filter': 'url(#colorMat)'})
    writer.element('path', attrib={'d': dpath, 'fill': f'url(#GR{self._n_gradients:x}_0)', 'shape-rendering': 'crispEdges'})
    writer.element('path', attrib={'d': dpath, 'fill': f'url(#GR{self._n_gradients:x}_1)', 'filter': 'url(#colorAdd)', 'shape-rendering': 'crispEdges'})
    writer.element('path', attrib={'d': dpath, 'fill': f'url(#GR{self._n_gradients:x}_2)', 'filter': 'url(#colorAdd)', 'shape-rendering': 'crispEdges'})
    writer.end('g')
    self._n_gradients += 1