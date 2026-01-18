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
def JM_set_field_type(doc, obj, type):
    """
    Set the field type
    """
    setbits = 0
    clearbits = 0
    typename = None
    if type == mupdf.PDF_WIDGET_TYPE_BUTTON:
        typename = PDF_NAME('Btn')
        setbits = mupdf.PDF_BTN_FIELD_IS_PUSHBUTTON
    elif type == mupdf.PDF_WIDGET_TYPE_RADIOBUTTON:
        typename = PDF_NAME('Btn')
        clearbits = mupdf.PDF_BTN_FIELD_IS_PUSHBUTTON
        setbits = mupdf.PDF_BTN_FIELD_IS_RADIO
    elif type == mupdf.PDF_WIDGET_TYPE_CHECKBOX:
        typename = PDF_NAME('Btn')
        clearbits = mupdf.PDF_BTN_FIELD_IS_PUSHBUTTON | mupdf.PDF_BTN_FIELD_IS_RADIO
    elif type == mupdf.PDF_WIDGET_TYPE_TEXT:
        typename = PDF_NAME('Tx')
    elif type == mupdf.PDF_WIDGET_TYPE_LISTBOX:
        typename = PDF_NAME('Ch')
        clearbits = mupdf.PDF_CH_FIELD_IS_COMBO
    elif type == mupdf.PDF_WIDGET_TYPE_COMBOBOX:
        typename = PDF_NAME('Ch')
        setbits = mupdf.PDF_CH_FIELD_IS_COMBO
    elif type == mupdf.PDF_WIDGET_TYPE_SIGNATURE:
        typename = PDF_NAME('Sig')
    if typename:
        mupdf.pdf_dict_put(obj, PDF_NAME('FT'), typename)
    if setbits != 0 or clearbits != 0:
        bits = mupdf.pdf_dict_get_int(obj, PDF_NAME('Ff'))
        bits &= ~clearbits
        bits |= setbits
        mupdf.pdf_dict_put_int(obj, PDF_NAME('Ff'), bits)