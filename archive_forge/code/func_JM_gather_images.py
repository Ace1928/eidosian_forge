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
def JM_gather_images(doc: mupdf.PdfDocument, dict_: mupdf.PdfObj, imagelist, stream_xref: int):
    """
    Store info of an image in Python list
    """
    rc = 1
    n = mupdf.pdf_dict_len(dict_)
    for i in range(n):
        refname = mupdf.pdf_dict_get_key(dict_, i)
        imagedict = mupdf.pdf_dict_get_val(dict_, i)
        if not mupdf.pdf_is_dict(imagedict):
            mupdf.fz_warn(f"'{mupdf.pdf_to_name(refname)}' is no image dict ({mupdf.pdf_to_num(imagedict)} 0 R)")
            continue
        type_ = mupdf.pdf_dict_get(imagedict, PDF_NAME('Subtype'))
        if not mupdf.pdf_name_eq(type_, PDF_NAME('Image')):
            continue
        xref = mupdf.pdf_to_num(imagedict)
        gen = 0
        smask = mupdf.pdf_dict_geta(imagedict, PDF_NAME('SMask'), PDF_NAME('Mask'))
        if smask.m_internal:
            gen = mupdf.pdf_to_num(smask)
        filter_ = mupdf.pdf_dict_geta(imagedict, PDF_NAME('Filter'), PDF_NAME('F'))
        if mupdf.pdf_is_array(filter_):
            filter_ = mupdf.pdf_array_get(filter_, 0)
        altcs = mupdf.PdfObj(0)
        cs = mupdf.pdf_dict_geta(imagedict, PDF_NAME('ColorSpace'), PDF_NAME('CS'))
        if mupdf.pdf_is_array(cs):
            cses = cs
            cs = mupdf.pdf_array_get(cses, 0)
            if mupdf.pdf_name_eq(cs, PDF_NAME('DeviceN')) or mupdf.pdf_name_eq(cs, PDF_NAME('Separation')):
                altcs = mupdf.pdf_array_get(cses, 2)
                if mupdf.pdf_is_array(altcs):
                    altcs = mupdf.pdf_array_get(altcs, 0)
        width = mupdf.pdf_dict_geta(imagedict, PDF_NAME('Width'), PDF_NAME('W'))
        height = mupdf.pdf_dict_geta(imagedict, PDF_NAME('Height'), PDF_NAME('H'))
        bpc = mupdf.pdf_dict_geta(imagedict, PDF_NAME('BitsPerComponent'), PDF_NAME('BPC'))
        entry = (xref, gen, mupdf.pdf_to_int(width), mupdf.pdf_to_int(height), mupdf.pdf_to_int(bpc), JM_EscapeStrFromStr(mupdf.pdf_to_name(cs)), JM_EscapeStrFromStr(mupdf.pdf_to_name(altcs)), JM_EscapeStrFromStr(mupdf.pdf_to_name(refname)), JM_EscapeStrFromStr(mupdf.pdf_to_name(filter_)), stream_xref)
        imagelist.append(entry)
    return rc