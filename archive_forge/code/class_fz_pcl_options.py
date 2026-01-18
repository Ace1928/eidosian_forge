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
class fz_pcl_options(object):
    """	PCL output"""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    features = property(_mupdf.fz_pcl_options_features_get, _mupdf.fz_pcl_options_features_set)
    odd_page_init = property(_mupdf.fz_pcl_options_odd_page_init_get, _mupdf.fz_pcl_options_odd_page_init_set)
    even_page_init = property(_mupdf.fz_pcl_options_even_page_init_get, _mupdf.fz_pcl_options_even_page_init_set)
    tumble = property(_mupdf.fz_pcl_options_tumble_get, _mupdf.fz_pcl_options_tumble_set)
    duplex_set = property(_mupdf.fz_pcl_options_duplex_set_get, _mupdf.fz_pcl_options_duplex_set_set)
    duplex = property(_mupdf.fz_pcl_options_duplex_get, _mupdf.fz_pcl_options_duplex_set)
    paper_size = property(_mupdf.fz_pcl_options_paper_size_get, _mupdf.fz_pcl_options_paper_size_set)
    manual_feed_set = property(_mupdf.fz_pcl_options_manual_feed_set_get, _mupdf.fz_pcl_options_manual_feed_set_set)
    manual_feed = property(_mupdf.fz_pcl_options_manual_feed_get, _mupdf.fz_pcl_options_manual_feed_set)
    media_position_set = property(_mupdf.fz_pcl_options_media_position_set_get, _mupdf.fz_pcl_options_media_position_set_set)
    media_position = property(_mupdf.fz_pcl_options_media_position_get, _mupdf.fz_pcl_options_media_position_set)
    orientation = property(_mupdf.fz_pcl_options_orientation_get, _mupdf.fz_pcl_options_orientation_set)
    page_count = property(_mupdf.fz_pcl_options_page_count_get, _mupdf.fz_pcl_options_page_count_set)

    def __init__(self):
        _mupdf.fz_pcl_options_swiginit(self, _mupdf.new_fz_pcl_options())
    __swig_destroy__ = _mupdf.delete_fz_pcl_options