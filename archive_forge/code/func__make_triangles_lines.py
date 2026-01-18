from __future__ import absolute_import
from __future__ import division
import pythreejs
import os
import time
import warnings
import tempfile
import uuid
import base64
from io import BytesIO as StringIO
import six
import numpy as np
import PIL.Image
import matplotlib.style
import ipywidgets
import IPython
from IPython.display import display
import ipyvolume as ipv
import ipyvolume.embed
from ipyvolume import utils
from . import ui
def _make_triangles_lines(shape, wrapx=False, wrapy=False):
    """Transform rectangular regular grid into triangles.

    :param x: {x2d}
    :param y: {y2d}
    :param z: {z2d}
    :param bool wrapx: when True, the x direction is assumed to wrap, and polygons are drawn between the end end begin points
    :param bool wrapy: simular for the y coordinate
    :return: triangles and lines used to plot Mesh
    """
    nx, ny = shape
    mx = nx if wrapx else nx - 1
    my = ny if wrapy else ny - 1
    '\n    create all pair of indices (i,j) of the rectangular grid\n    minus last row if wrapx = False => mx\n    minus last column if wrapy = False => my\n    |  (0,0)   ...   (0,j)    ...   (0,my-1)  |\n    |    .      .      .       .       .      |\n    |  (i,0)   ...   (i,j)    ...   (i,my-1)  |\n    |    .      .      .       .       .      |\n    |(mx-1,0)  ...  (mx-1,j)  ... (mx-1,my-1) |\n    '
    i, j = np.mgrid[0:mx, 0:my]
    '\n    collapsed i and j in one dimensional array, row-major order\n    ex :\n    array([[0,  1,  2],     =>   array([0, 1, 2, 3, *4*, 5])\n           [3, *4*, 5]])\n    if we want vertex 4 at (i=1,j=1) we must transform it in i*ny+j = 4\n    '
    i, j = (np.ravel(i), np.ravel(j))
    "\n    Let's go for the triangles :\n        (i,j)    -  (i,j+1)   -> y dir\n        (i+1,j)  - (i+1,j+1)\n          |\n          v\n        x dir\n\n    in flatten coordinates:\n        i*ny+j     -  i*ny+j+1\n        (i+1)*ny+j -  (i+1)*ny+j+1\n    "
    t1 = (i * ny + j, (i + 1) % nx * ny + j, (i + 1) % nx * ny + (j + 1) % ny)
    t2 = (i * ny + j, (i + 1) % nx * ny + (j + 1) % ny, i * ny + (j + 1) % ny)
    '\n        %nx and %ny are used for wrapx and wrapy :\n        if (i+1)=nx => (i+1)%nx=0 => close mesh in x direction\n        if (j+1)=ny => (j+1)%ny=0 => close mesh in y direction\n    '
    nt = len(t1[0])
    triangles = np.zeros((nt * 2, 3), dtype=np.uint32)
    triangles[0::2, 0], triangles[0::2, 1], triangles[0::2, 2] = t1
    triangles[1::2, 0], triangles[1::2, 1], triangles[1::2, 2] = t2
    lines = np.zeros((nt * 4, 2), dtype=np.uint32)
    lines[::4, 0], lines[::4, 1] = t1[:2]
    lines[1::4, 0], lines[1::4, 1] = (t1[0], t2[2])
    lines[2::4, 0], lines[2::4, 1] = t2[2:0:-1]
    lines[3::4, 0], lines[3::4, 1] = (t1[1], t2[1])
    return (triangles, lines)