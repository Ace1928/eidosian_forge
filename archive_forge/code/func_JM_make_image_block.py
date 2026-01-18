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
def JM_make_image_block(block, block_dict):
    image = block.i_image()
    n = mupdf.fz_colorspace_n(image.colorspace())
    w = image.w()
    h = image.h()
    type_ = mupdf.FZ_IMAGE_UNKNOWN
    ll_fz_compressed_buffer = mupdf.ll_fz_compressed_image_buffer(image.m_internal)
    if ll_fz_compressed_buffer:
        type_ = ll_fz_compressed_buffer.params.type
    if type_ < mupdf.FZ_IMAGE_BMP or type_ == mupdf.FZ_IMAGE_JBIG2:
        type_ = mupdf.FZ_IMAGE_UNKNOWN
    bytes_ = None
    if ll_fz_compressed_buffer and type_ != mupdf.FZ_IMAGE_UNKNOWN:
        buf = mupdf.FzBuffer(mupdf.ll_fz_keep_buffer(ll_fz_compressed_buffer.buffer))
        ext = JM_image_extension(type_)
    else:
        buf = mupdf.fz_new_buffer_from_image_as_png(image, mupdf.FzColorParams())
        ext = 'png'
    bytes_ = JM_BinFromBuffer(buf)
    block_dict[dictkey_width] = w
    block_dict[dictkey_height] = h
    block_dict[dictkey_ext] = ext
    block_dict[dictkey_colorspace] = n
    block_dict[dictkey_xres] = image.xres()
    block_dict[dictkey_yres] = image.yres()
    block_dict[dictkey_bpc] = image.bpc()
    block_dict[dictkey_matrix] = JM_py_from_matrix(block.i_transform())
    block_dict[dictkey_size] = len(bytes_)
    block_dict[dictkey_image] = bytes_