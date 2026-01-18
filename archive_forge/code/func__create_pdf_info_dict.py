import codecs
from datetime import timezone
from datetime import datetime
from enum import Enum
from functools import total_ordering
from io import BytesIO
import itertools
import logging
import math
import os
import string
import struct
import sys
import time
import types
import warnings
import zlib
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import _api, _text_helpers, _type1font, cbook, dviread
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.figure import Figure
from matplotlib.font_manager import get_font, fontManager as _fontManager
from matplotlib._afm import AFM
from matplotlib.ft2font import (FIXED_WIDTH, ITALIC, LOAD_NO_SCALE,
from matplotlib.transforms import Affine2D, BboxBase
from matplotlib.path import Path
from matplotlib.dates import UTC
from matplotlib import _path
from . import _backend_pdf_ps
def _create_pdf_info_dict(backend, metadata):
    """
    Create a PDF infoDict based on user-supplied metadata.

    A default ``Creator``, ``Producer``, and ``CreationDate`` are added, though
    the user metadata may override it. The date may be the current time, or a
    time set by the ``SOURCE_DATE_EPOCH`` environment variable.

    Metadata is verified to have the correct keys and their expected types. Any
    unknown keys/types will raise a warning.

    Parameters
    ----------
    backend : str
        The name of the backend to use in the Producer value.

    metadata : dict[str, Union[str, datetime, Name]]
        A dictionary of metadata supplied by the user with information
        following the PDF specification, also defined in
        `~.backend_pdf.PdfPages` below.

        If any value is *None*, then the key will be removed. This can be used
        to remove any pre-defined values.

    Returns
    -------
    dict[str, Union[str, datetime, Name]]
        A validated dictionary of metadata.
    """
    source_date_epoch = os.getenv('SOURCE_DATE_EPOCH')
    if source_date_epoch:
        source_date = datetime.fromtimestamp(int(source_date_epoch), timezone.utc)
        source_date = source_date.replace(tzinfo=UTC)
    else:
        source_date = datetime.today()
    info = {'Creator': f'Matplotlib v{mpl.__version__}, https://matplotlib.org', 'Producer': f'Matplotlib {backend} backend v{mpl.__version__}', 'CreationDate': source_date, **metadata}
    info = {k: v for k, v in info.items() if v is not None}

    def is_string_like(x):
        return isinstance(x, str)
    is_string_like.text_for_warning = 'an instance of str'

    def is_date(x):
        return isinstance(x, datetime)
    is_date.text_for_warning = 'an instance of datetime.datetime'

    def check_trapped(x):
        if isinstance(x, Name):
            return x.name in (b'True', b'False', b'Unknown')
        else:
            return x in ('True', 'False', 'Unknown')
    check_trapped.text_for_warning = 'one of {"True", "False", "Unknown"}'
    keywords = {'Title': is_string_like, 'Author': is_string_like, 'Subject': is_string_like, 'Keywords': is_string_like, 'Creator': is_string_like, 'Producer': is_string_like, 'CreationDate': is_date, 'ModDate': is_date, 'Trapped': check_trapped}
    for k in info:
        if k not in keywords:
            _api.warn_external(f'Unknown infodict keyword: {k!r}. Must be one of {set(keywords)!r}.')
        elif not keywords[k](info[k]):
            _api.warn_external(f'Bad value for infodict keyword {k}. Got {info[k]!r} which is not {keywords[k].text_for_warning}.')
    if 'Trapped' in info:
        info['Trapped'] = Name(info['Trapped'])
    return info