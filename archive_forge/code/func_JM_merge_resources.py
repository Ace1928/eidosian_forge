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
def JM_merge_resources(page, temp_res):
    """
    Merge the /Resources object created by a text pdf device into the page.
    The device may have created multiple /ExtGState/Alp? and /Font/F? objects.
    These need to be renamed (renumbered) to not overwrite existing page
    objects from previous executions.
    Returns the next available numbers n, m for objects /Alp<n>, /F<m>.
    """
    resources = mupdf.pdf_dict_get(page.obj(), PDF_NAME('Resources'))
    main_extg = mupdf.pdf_dict_get(resources, PDF_NAME('ExtGState'))
    main_fonts = mupdf.pdf_dict_get(resources, PDF_NAME('Font'))
    temp_extg = mupdf.pdf_dict_get(temp_res, PDF_NAME('ExtGState'))
    temp_fonts = mupdf.pdf_dict_get(temp_res, PDF_NAME('Font'))
    max_alp = -1
    max_fonts = -1
    if mupdf.pdf_is_dict(temp_extg):
        n = mupdf.pdf_dict_len(temp_extg)
        if mupdf.pdf_is_dict(main_extg):
            for i in range(mupdf.pdf_dict_len(main_extg)):
                alp = mupdf.pdf_to_name(mupdf.pdf_dict_get_key(main_extg, i))
                if not alp.startswith('Alp'):
                    continue
                j = mupdf.fz_atoi(alp[3:])
                if j > max_alp:
                    max_alp = j
        else:
            main_extg = mupdf.pdf_dict_put_dict(resources, PDF_NAME('ExtGState'), n)
        max_alp += 1
        for i in range(n):
            alp = mupdf.pdf_to_name(mupdf.pdf_dict_get_key(temp_extg, i))
            j = mupdf.fz_atoi(alp[3:]) + max_alp
            text = f'Alp{j}'
            val = mupdf.pdf_dict_get_val(temp_extg, i)
            mupdf.pdf_dict_puts(main_extg, text, val)
    if mupdf.pdf_is_dict(main_fonts):
        for i in range(mupdf.pdf_dict_len(main_fonts)):
            font = mupdf.pdf_to_name(mupdf.pdf_dict_get_key(main_fonts, i))
            if not font.startswith('F'):
                continue
            j = mupdf.fz_atoi(font[1:])
            if j > max_fonts:
                max_fonts = j
    else:
        main_fonts = mupdf.pdf_dict_put_dict(resources, PDF_NAME('Font'), 2)
    max_fonts += 1
    for i in range(mupdf.pdf_dict_len(temp_fonts)):
        font = mupdf.pdf_to_name(mupdf.pdf_dict_get_key(temp_fonts, i))
        j = mupdf.fz_atoi(font[1:]) + max_fonts
        text = f'F{j}'
        val = mupdf.pdf_dict_get_val(temp_fonts, i)
        mupdf.pdf_dict_puts(main_fonts, text, val)
    return (max_alp, max_fonts)