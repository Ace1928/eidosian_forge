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
def JM_field_type_text(wtype):
    """
    String from widget type
    """
    if wtype == mupdf.PDF_WIDGET_TYPE_BUTTON:
        return 'Button'
    if wtype == mupdf.PDF_WIDGET_TYPE_CHECKBOX:
        return 'CheckBox'
    if wtype == mupdf.PDF_WIDGET_TYPE_RADIOBUTTON:
        return 'RadioButton'
    if wtype == mupdf.PDF_WIDGET_TYPE_TEXT:
        return 'Text'
    if wtype == mupdf.PDF_WIDGET_TYPE_LISTBOX:
        return 'ListBox'
    if wtype == mupdf.PDF_WIDGET_TYPE_COMBOBOX:
        return 'ComboBox'
    if wtype == mupdf.PDF_WIDGET_TYPE_SIGNATURE:
        return 'Signature'
    return 'unknown'