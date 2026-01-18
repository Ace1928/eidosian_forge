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
class fz_pwg_options(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    media_class = property(_mupdf.fz_pwg_options_media_class_get, _mupdf.fz_pwg_options_media_class_set)
    media_color = property(_mupdf.fz_pwg_options_media_color_get, _mupdf.fz_pwg_options_media_color_set)
    media_type = property(_mupdf.fz_pwg_options_media_type_get, _mupdf.fz_pwg_options_media_type_set)
    output_type = property(_mupdf.fz_pwg_options_output_type_get, _mupdf.fz_pwg_options_output_type_set)
    advance_distance = property(_mupdf.fz_pwg_options_advance_distance_get, _mupdf.fz_pwg_options_advance_distance_set)
    advance_media = property(_mupdf.fz_pwg_options_advance_media_get, _mupdf.fz_pwg_options_advance_media_set)
    collate = property(_mupdf.fz_pwg_options_collate_get, _mupdf.fz_pwg_options_collate_set)
    cut_media = property(_mupdf.fz_pwg_options_cut_media_get, _mupdf.fz_pwg_options_cut_media_set)
    duplex = property(_mupdf.fz_pwg_options_duplex_get, _mupdf.fz_pwg_options_duplex_set)
    insert_sheet = property(_mupdf.fz_pwg_options_insert_sheet_get, _mupdf.fz_pwg_options_insert_sheet_set)
    jog = property(_mupdf.fz_pwg_options_jog_get, _mupdf.fz_pwg_options_jog_set)
    leading_edge = property(_mupdf.fz_pwg_options_leading_edge_get, _mupdf.fz_pwg_options_leading_edge_set)
    manual_feed = property(_mupdf.fz_pwg_options_manual_feed_get, _mupdf.fz_pwg_options_manual_feed_set)
    media_position = property(_mupdf.fz_pwg_options_media_position_get, _mupdf.fz_pwg_options_media_position_set)
    media_weight = property(_mupdf.fz_pwg_options_media_weight_get, _mupdf.fz_pwg_options_media_weight_set)
    mirror_print = property(_mupdf.fz_pwg_options_mirror_print_get, _mupdf.fz_pwg_options_mirror_print_set)
    negative_print = property(_mupdf.fz_pwg_options_negative_print_get, _mupdf.fz_pwg_options_negative_print_set)
    num_copies = property(_mupdf.fz_pwg_options_num_copies_get, _mupdf.fz_pwg_options_num_copies_set)
    orientation = property(_mupdf.fz_pwg_options_orientation_get, _mupdf.fz_pwg_options_orientation_set)
    output_face_up = property(_mupdf.fz_pwg_options_output_face_up_get, _mupdf.fz_pwg_options_output_face_up_set)
    PageSize = property(_mupdf.fz_pwg_options_PageSize_get, _mupdf.fz_pwg_options_PageSize_set)
    separations = property(_mupdf.fz_pwg_options_separations_get, _mupdf.fz_pwg_options_separations_set)
    tray_switch = property(_mupdf.fz_pwg_options_tray_switch_get, _mupdf.fz_pwg_options_tray_switch_set)
    tumble = property(_mupdf.fz_pwg_options_tumble_get, _mupdf.fz_pwg_options_tumble_set)
    media_type_num = property(_mupdf.fz_pwg_options_media_type_num_get, _mupdf.fz_pwg_options_media_type_num_set)
    compression = property(_mupdf.fz_pwg_options_compression_get, _mupdf.fz_pwg_options_compression_set)
    row_count = property(_mupdf.fz_pwg_options_row_count_get, _mupdf.fz_pwg_options_row_count_set)
    row_feed = property(_mupdf.fz_pwg_options_row_feed_get, _mupdf.fz_pwg_options_row_feed_set)
    row_step = property(_mupdf.fz_pwg_options_row_step_get, _mupdf.fz_pwg_options_row_step_set)
    rendering_intent = property(_mupdf.fz_pwg_options_rendering_intent_get, _mupdf.fz_pwg_options_rendering_intent_set)
    page_size_name = property(_mupdf.fz_pwg_options_page_size_name_get, _mupdf.fz_pwg_options_page_size_name_set)

    def __init__(self):
        _mupdf.fz_pwg_options_swiginit(self, _mupdf.new_fz_pwg_options())
    __swig_destroy__ = _mupdf.delete_fz_pwg_options