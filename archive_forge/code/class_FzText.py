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
class FzText(object):
    """ Wrapper class for struct `fz_text`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def fz_bound_text(self, stroke, ctm):
        """
        Class-aware wrapper for `::fz_bound_text()`.
        	Find the bounds of a given text object.

        	text: The text object to find the bounds of.

        	stroke: Pointer to the stroke attributes (for stroked
        	text), or NULL (for filled text).

        	ctm: The matrix in use.

        	r: pointer to storage for the bounds.

        	Returns a pointer to r, which is updated to contain the
        	bounding box for the text object.
        """
        return _mupdf.FzText_fz_bound_text(self, stroke, ctm)

    def fz_show_glyph(self, font, trm, glyph, unicode, wmode, bidi_level, markup_dir, language):
        """
        Class-aware wrapper for `::fz_show_glyph()`.
        	Add a glyph/unicode value to a text object.

        	text: Text object to add to.

        	font: The font the glyph should be added in.

        	trm: The transform to use for the glyph.

        	glyph: The glyph id to add.

        	unicode: The unicode character for the glyph.

        	cid: The CJK CID value or raw character code.

        	wmode: 1 for vertical mode, 0 for horizontal.

        	bidi_level: The bidirectional level for this glyph.

        	markup_dir: The direction of the text as specified in the
        	markup.

        	language: The language in use (if known, 0 otherwise)
        	(e.g. FZ_LANG_zh_Hans).

        	Throws exception on failure to allocate.
        """
        return _mupdf.FzText_fz_show_glyph(self, font, trm, glyph, unicode, wmode, bidi_level, markup_dir, language)

    def fz_show_glyph_aux(self, font, trm, glyph, unicode, cid, wmode, bidi_level, markup_dir, lang):
        """ Class-aware wrapper for `::fz_show_glyph_aux()`."""
        return _mupdf.FzText_fz_show_glyph_aux(self, font, trm, glyph, unicode, cid, wmode, bidi_level, markup_dir, lang)

    def fz_show_string(self, font, trm, s, wmode, bidi_level, markup_dir, language):
        """
        Class-aware wrapper for `::fz_show_string()`.
        	Add a UTF8 string to a text object.

        	text: Text object to add to.

        	font: The font the string should be added in.

        	trm: The transform to use.

        	s: The utf-8 string to add.

        	wmode: 1 for vertical mode, 0 for horizontal.

        	bidi_level: The bidirectional level for this glyph.

        	markup_dir: The direction of the text as specified in the markup.

        	language: The language in use (if known, 0 otherwise)
        		(e.g. FZ_LANG_zh_Hans).

        	Returns the transform updated with the advance width of the
        	string.
        """
        return _mupdf.FzText_fz_show_string(self, font, trm, s, wmode, bidi_level, markup_dir, language)

    def __init__(self, *args):
        """
        *Overload 1:*
         == Constructors.  Constructor using `fz_new_text()`.
        		Create a new empty fz_text object.

        		Throws exception on failure to allocate.


        |

        *Overload 2:*
         Copy constructor using `fz_keep_text()`.

        |

        *Overload 3:*
         Constructor using raw copy of pre-existing `::fz_text`.
        """
        _mupdf.FzText_swiginit(self, _mupdf.new_FzText(*args))
    __swig_destroy__ = _mupdf.delete_FzText

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.FzText_m_internal_value(self)
    m_internal = property(_mupdf.FzText_m_internal_get, _mupdf.FzText_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.FzText_s_num_instances_get, _mupdf.FzText_s_num_instances_set)