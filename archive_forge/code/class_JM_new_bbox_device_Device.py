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
class JM_new_bbox_device_Device(mupdf.FzDevice2):

    def __init__(self, result, layers):
        super().__init__()
        self.result = result
        self.layers = layers
        self.use_virtual_fill_path()
        self.use_virtual_stroke_path()
        self.use_virtual_fill_text()
        self.use_virtual_stroke_text()
        self.use_virtual_ignore_text()
        self.use_virtual_fill_shade()
        self.use_virtual_fill_image()
        self.use_virtual_fill_image_mask()
        self.use_virtual_begin_layer()
        self.use_virtual_end_layer()
    begin_layer = jm_lineart_begin_layer
    end_layer = jm_lineart_end_layer
    fill_path = jm_bbox_fill_path
    stroke_path = jm_bbox_stroke_path
    fill_text = jm_bbox_fill_text
    stroke_text = jm_bbox_stroke_text
    ignore_text = jm_bbox_ignore_text
    fill_shade = jm_bbox_fill_shade
    fill_image = jm_bbox_fill_image
    fill_image_mask = jm_bbox_fill_image_mask