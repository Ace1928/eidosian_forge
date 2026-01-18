from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
class FzPdfocrOptions(object):
    """
    Wrapper class for struct `fz_pdfocr_options`.
    PDFOCR output
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_parse_pdfocr_options(self, args):
        """
        Class-aware wrapper for `::fz_parse_pdfocr_options()`.
        	Parse PDFOCR options.

        	Currently defined options and values are as follows:

        		compression=none: No compression
        		compression=flate: Flate compression
        		strip-height=n: Strip height (default 16)
        		ocr-language=<lang>: OCR Language (default eng)
        		ocr-datadir=<datadir>: OCR data path (default rely on TESSDATA_PREFIX)
        """
        return _mupdf.FzPdfocrOptions_fz_parse_pdfocr_options(self, args)

    def language_set2(self, language):
        """ Copies <language> into this->language, truncating if necessary."""
        return _mupdf.FzPdfocrOptions_language_set2(self, language)

    def datadir_set2(self, datadir):
        """ Copies <datadir> into this->datadir, truncating if necessary."""
        return _mupdf.FzPdfocrOptions_datadir_set2(self, datadir)

    def __init__(self, *args):
        """
        *Overload 1:*
        We use default copy constructor and operator=.  Default constructor, sets each member to default value.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::fz_pdfocr_options`.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::fz_pdfocr_options`.
        """
        _mupdf.FzPdfocrOptions_swiginit(self, _mupdf.new_FzPdfocrOptions(*args))

    def internal(self, *args):
        """
        *Overload 1:*
        Access as underlying struct.

        |

        *Overload 2:*
        Access as underlying struct.
        """
        return _mupdf.FzPdfocrOptions_internal(self, *args)
    __swig_destroy__ = _mupdf.delete_FzPdfocrOptions
    compress = property(_mupdf.FzPdfocrOptions_compress_get, _mupdf.FzPdfocrOptions_compress_set)
    strip_height = property(_mupdf.FzPdfocrOptions_strip_height_get, _mupdf.FzPdfocrOptions_strip_height_set)
    language = property(_mupdf.FzPdfocrOptions_language_get, _mupdf.FzPdfocrOptions_language_set)
    datadir = property(_mupdf.FzPdfocrOptions_datadir_get, _mupdf.FzPdfocrOptions_datadir_set)
    page_count = property(_mupdf.FzPdfocrOptions_page_count_get, _mupdf.FzPdfocrOptions_page_count_set)
    s_num_instances = property(_mupdf.FzPdfocrOptions_s_num_instances_get, _mupdf.FzPdfocrOptions_s_num_instances_set)

    def to_string(self):
        """ Returns string containing our members, labelled and inside (...), using operator<<."""
        return _mupdf.FzPdfocrOptions_to_string(self)

    def __eq__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzPdfocrOptions___eq__(self, rhs)

    def __ne__(self, rhs):
        """ Comparison method."""
        return _mupdf.FzPdfocrOptions___ne__(self, rhs)