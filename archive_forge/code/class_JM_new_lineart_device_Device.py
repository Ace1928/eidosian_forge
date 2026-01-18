import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
class JM_new_lineart_device_Device(mupdf.FzDevice2):
    """
    LINEART device for Python method Page.get_cdrawings()
    """

    def __init__(self, out, clips, method):
        super().__init__()
        self.use_virtual_fill_path()
        self.use_virtual_stroke_path()
        self.use_virtual_clip_path()
        self.use_virtual_clip_image_mask()
        self.use_virtual_clip_stroke_path()
        self.use_virtual_clip_stroke_text()
        self.use_virtual_clip_text()
        self.use_virtual_fill_text
        self.use_virtual_stroke_text
        self.use_virtual_ignore_text
        self.use_virtual_fill_shade()
        self.use_virtual_fill_image()
        self.use_virtual_fill_image_mask()
        self.use_virtual_pop_clip()
        self.use_virtual_begin_group()
        self.use_virtual_end_group()
        self.use_virtual_begin_layer()
        self.use_virtual_end_layer()
        self.out = out
        self.seqno = 0
        self.depth = 0
        self.clips = clips
        self.method = method
        self.scissors = None
        self.layer_name = ''
        self.pathrect = None
        self.linewidth = 0
        self.ptm = mupdf.FzMatrix()
        self.ctm = mupdf.FzMatrix()
        self.rot = mupdf.FzMatrix()
        self.lastpoint = mupdf.FzPoint()
        self.firstpoint = mupdf.FzPoint()
        self.havemove = 0
        self.pathrect = mupdf.FzRect()
        self.pathfactor = 0
        self.linecount = 0
        self.path_type = 0
    fill_path = jm_lineart_fill_path
    stroke_path = jm_lineart_stroke_path
    clip_image_mask = jm_lineart_clip_image_mask
    clip_path = jm_lineart_clip_path
    clip_stroke_path = jm_lineart_clip_stroke_path
    clip_text = jm_lineart_clip_text
    clip_stroke_text = jm_lineart_clip_stroke_text
    fill_text = jm_increase_seqno
    stroke_text = jm_increase_seqno
    ignore_text = jm_increase_seqno
    fill_shade = jm_increase_seqno
    fill_image = jm_increase_seqno
    fill_image_mask = jm_increase_seqno
    pop_clip = jm_lineart_pop_clip
    begin_group = jm_lineart_begin_group
    end_group = jm_lineart_end_group
    begin_layer = jm_lineart_begin_layer
    end_layer = jm_lineart_end_layer