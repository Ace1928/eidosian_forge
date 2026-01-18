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
def JM_print_stext_page_as_text(res, page):
    """
    Plain text output. An identical copy of fz_print_stext_page_as_text,
    but lines within a block are concatenated by space instead a new-line
    character (which else leads to 2 new-lines).
    """
    if 1 and g_use_extra:
        return extra.JM_print_stext_page_as_text(res, page)
    assert isinstance(res, mupdf.FzBuffer)
    assert isinstance(page, mupdf.FzStextPage)
    rect = mupdf.FzRect(page.m_internal.mediabox)
    last_char = 0
    n_blocks = 0
    n_lines = 0
    n_chars = 0
    for n_blocks2, block in enumerate(page):
        if block.m_internal.type == mupdf.FZ_STEXT_BLOCK_TEXT:
            for n_lines2, line in enumerate(block):
                for n_chars2, ch in enumerate(line):
                    pass
                n_chars += n_chars2
            n_lines += n_lines2
        n_blocks += n_blocks2
    for block in page:
        if block.m_internal.type == mupdf.FZ_STEXT_BLOCK_TEXT:
            for line in block:
                last_char = 0
                for ch in line:
                    chbbox = JM_char_bbox(line, ch)
                    if mupdf.fz_is_infinite_rect(rect) or JM_rects_overlap(rect, chbbox):
                        last_char = ch.m_internal.c
                        JM_append_rune(res, last_char)
                if last_char != 10 and last_char > 0:
                    mupdf.fz_append_string(res, '\n')