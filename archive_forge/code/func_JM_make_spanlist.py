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
def JM_make_spanlist(line_dict, line, raw, buff, tp_rect):
    if g_use_extra:
        return extra.JM_make_spanlist(line_dict, line, raw, buff, tp_rect)
    char_list = None
    span_list = []
    mupdf.fz_clear_buffer(buff)
    span_rect = mupdf.FzRect(mupdf.FzRect.Fixed_EMPTY)
    line_rect = mupdf.FzRect(mupdf.FzRect.Fixed_EMPTY)

    class char_style:

        def __init__(self, rhs=None):
            if rhs:
                self.size = rhs.size
                self.flags = rhs.flags
                self.font = rhs.font
                self.color = rhs.color
                self.asc = rhs.asc
                self.desc = rhs.desc
            else:
                self.size = -1
                self.flags = -1
                self.font = ''
                self.color = -1
                self.asc = 0
                self.desc = 0

        def __str__(self):
            return f'{self.size} {self.flags} {self.font} {self.color} {self.asc} {self.desc}'
    old_style = char_style()
    style = char_style()
    span = None
    span_origin = None
    for ch in line:
        r = JM_char_bbox(line, ch)
        if not JM_rects_overlap(tp_rect, r) and (not mupdf.fz_is_infinite_rect(tp_rect)):
            continue
        flags = JM_char_font_flags(mupdf.FzFont(mupdf.ll_fz_keep_font(ch.m_internal.font)), line, ch)
        origin = mupdf.FzPoint(ch.m_internal.origin)
        style.size = ch.m_internal.size
        style.flags = flags
        style.font = JM_font_name(mupdf.FzFont(mupdf.ll_fz_keep_font(ch.m_internal.font)))
        style.color = ch.m_internal.color
        style.asc = JM_font_ascender(mupdf.FzFont(mupdf.ll_fz_keep_font(ch.m_internal.font)))
        style.desc = JM_font_descender(mupdf.FzFont(mupdf.ll_fz_keep_font(ch.m_internal.font)))
        if style.size != old_style.size or style.flags != old_style.flags or style.color != old_style.color or (style.font != old_style.font):
            if old_style.size >= 0:
                if raw:
                    span[dictkey_chars] = char_list
                    char_list = None
                else:
                    span[dictkey_text] = JM_EscapeStrFromBuffer(buff)
                    mupdf.fz_clear_buffer(buff)
                span[dictkey_origin] = JM_py_from_point(span_origin)
                span[dictkey_bbox] = JM_py_from_rect(span_rect)
                line_rect = mupdf.fz_union_rect(line_rect, span_rect)
                span_list.append(span)
                span = None
            span = dict()
            asc = style.asc
            desc = style.desc
            if style.asc < 0.001:
                asc = 0.9
                desc = -0.1
            span[dictkey_size] = style.size
            span[dictkey_flags] = style.flags
            span[dictkey_font] = JM_EscapeStrFromStr(style.font)
            span[dictkey_color] = style.color
            span['ascender'] = asc
            span['descender'] = desc
            old_style = char_style(style)
            span_rect = r
            span_origin = origin
        span_rect = mupdf.fz_union_rect(span_rect, r)
        if raw:
            char_dict = dict()
            char_dict[dictkey_origin] = JM_py_from_point(ch.m_internal.origin)
            char_dict[dictkey_bbox] = JM_py_from_rect(r)
            char_dict[dictkey_c] = chr(ch.m_internal.c)
            if char_list is None:
                char_list = []
            char_list.append(char_dict)
        else:
            JM_append_rune(buff, ch.m_internal.c)
    if span:
        if raw:
            span[dictkey_chars] = char_list
            char_list = None
        else:
            span[dictkey_text] = JM_EscapeStrFromBuffer(buff)
            mupdf.fz_clear_buffer(buff)
        span[dictkey_origin] = JM_py_from_point(span_origin)
        span[dictkey_bbox] = JM_py_from_rect(span_rect)
        if not mupdf.fz_is_empty_rect(span_rect):
            span_list.append(span)
            line_rect = mupdf.fz_union_rect(line_rect, span_rect)
        span = None
    if not mupdf.fz_is_empty_rect(line_rect):
        line_dict[dictkey_spans] = span_list
    else:
        line_dict[dictkey_spans] = span_list
    return line_rect