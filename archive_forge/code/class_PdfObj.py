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
class PdfObj(object):
    """ Wrapper class for struct `pdf_obj`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    @staticmethod
    def pdf_new_text_string(s):
        """ Class-aware wrapper for `::pdf_new_text_string()`."""
        return _mupdf.PdfObj_pdf_new_text_string(s)

    @staticmethod
    def pdf_new_dict(doc, initialcap):
        """ Class-aware wrapper for `::pdf_new_dict()`."""
        return _mupdf.PdfObj_pdf_new_dict(doc, initialcap)

    def pdf_array_contains(self, obj):
        """ Class-aware wrapper for `::pdf_array_contains()`."""
        return _mupdf.PdfObj_pdf_array_contains(self, obj)

    def pdf_array_delete(self, index):
        """ Class-aware wrapper for `::pdf_array_delete()`."""
        return _mupdf.PdfObj_pdf_array_delete(self, index)

    def pdf_array_find(self, obj):
        """ Class-aware wrapper for `::pdf_array_find()`."""
        return _mupdf.PdfObj_pdf_array_find(self, obj)

    def pdf_array_get(self, i):
        """ Class-aware wrapper for `::pdf_array_get()`."""
        return _mupdf.PdfObj_pdf_array_get(self, i)

    def pdf_array_get_bool(self, index):
        """ Class-aware wrapper for `::pdf_array_get_bool()`."""
        return _mupdf.PdfObj_pdf_array_get_bool(self, index)

    def pdf_array_get_int(self, index):
        """ Class-aware wrapper for `::pdf_array_get_int()`."""
        return _mupdf.PdfObj_pdf_array_get_int(self, index)

    def pdf_array_get_matrix(self, index):
        """ Class-aware wrapper for `::pdf_array_get_matrix()`."""
        return _mupdf.PdfObj_pdf_array_get_matrix(self, index)

    def pdf_array_get_name(self, index):
        """ Class-aware wrapper for `::pdf_array_get_name()`."""
        return _mupdf.PdfObj_pdf_array_get_name(self, index)

    def pdf_array_get_real(self, index):
        """ Class-aware wrapper for `::pdf_array_get_real()`."""
        return _mupdf.PdfObj_pdf_array_get_real(self, index)

    def pdf_array_get_rect(self, index):
        """ Class-aware wrapper for `::pdf_array_get_rect()`."""
        return _mupdf.PdfObj_pdf_array_get_rect(self, index)

    def pdf_array_get_string(self, index, sizep):
        """
        Class-aware wrapper for `::pdf_array_get_string()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_array_get_string(int index)` => `(const char *, size_t sizep)`
        """
        return _mupdf.PdfObj_pdf_array_get_string(self, index, sizep)

    def pdf_array_get_text_string(self, index):
        """ Class-aware wrapper for `::pdf_array_get_text_string()`."""
        return _mupdf.PdfObj_pdf_array_get_text_string(self, index)

    def pdf_array_insert(self, obj, index):
        """ Class-aware wrapper for `::pdf_array_insert()`."""
        return _mupdf.PdfObj_pdf_array_insert(self, obj, index)

    def pdf_array_len(self):
        """ Class-aware wrapper for `::pdf_array_len()`."""
        return _mupdf.PdfObj_pdf_array_len(self)

    def pdf_array_push(self, obj):
        """ Class-aware wrapper for `::pdf_array_push()`."""
        return _mupdf.PdfObj_pdf_array_push(self, obj)

    def pdf_array_push_array(self, initial):
        """ Class-aware wrapper for `::pdf_array_push_array()`."""
        return _mupdf.PdfObj_pdf_array_push_array(self, initial)

    def pdf_array_push_bool(self, x):
        """ Class-aware wrapper for `::pdf_array_push_bool()`."""
        return _mupdf.PdfObj_pdf_array_push_bool(self, x)

    def pdf_array_push_dict(self, initial):
        """ Class-aware wrapper for `::pdf_array_push_dict()`."""
        return _mupdf.PdfObj_pdf_array_push_dict(self, initial)

    def pdf_array_push_int(self, x):
        """ Class-aware wrapper for `::pdf_array_push_int()`."""
        return _mupdf.PdfObj_pdf_array_push_int(self, x)

    def pdf_array_push_name(self, x):
        """ Class-aware wrapper for `::pdf_array_push_name()`."""
        return _mupdf.PdfObj_pdf_array_push_name(self, x)

    def pdf_array_push_real(self, x):
        """ Class-aware wrapper for `::pdf_array_push_real()`."""
        return _mupdf.PdfObj_pdf_array_push_real(self, x)

    def pdf_array_push_string(self, x, n):
        """ Class-aware wrapper for `::pdf_array_push_string()`."""
        return _mupdf.PdfObj_pdf_array_push_string(self, x, n)

    def pdf_array_push_text_string(self, x):
        """ Class-aware wrapper for `::pdf_array_push_text_string()`."""
        return _mupdf.PdfObj_pdf_array_push_text_string(self, x)

    def pdf_array_put(self, i, obj):
        """ Class-aware wrapper for `::pdf_array_put()`."""
        return _mupdf.PdfObj_pdf_array_put(self, i, obj)

    def pdf_array_put_array(self, i, initial):
        """ Class-aware wrapper for `::pdf_array_put_array()`."""
        return _mupdf.PdfObj_pdf_array_put_array(self, i, initial)

    def pdf_array_put_bool(self, i, x):
        """ Class-aware wrapper for `::pdf_array_put_bool()`."""
        return _mupdf.PdfObj_pdf_array_put_bool(self, i, x)

    def pdf_array_put_dict(self, i, initial):
        """ Class-aware wrapper for `::pdf_array_put_dict()`."""
        return _mupdf.PdfObj_pdf_array_put_dict(self, i, initial)

    def pdf_array_put_int(self, i, x):
        """ Class-aware wrapper for `::pdf_array_put_int()`."""
        return _mupdf.PdfObj_pdf_array_put_int(self, i, x)

    def pdf_array_put_name(self, i, x):
        """ Class-aware wrapper for `::pdf_array_put_name()`."""
        return _mupdf.PdfObj_pdf_array_put_name(self, i, x)

    def pdf_array_put_real(self, i, x):
        """ Class-aware wrapper for `::pdf_array_put_real()`."""
        return _mupdf.PdfObj_pdf_array_put_real(self, i, x)

    def pdf_array_put_string(self, i, x, n):
        """ Class-aware wrapper for `::pdf_array_put_string()`."""
        return _mupdf.PdfObj_pdf_array_put_string(self, i, x, n)

    def pdf_array_put_text_string(self, i, x):
        """ Class-aware wrapper for `::pdf_array_put_text_string()`."""
        return _mupdf.PdfObj_pdf_array_put_text_string(self, i, x)

    def pdf_button_field_on_state(self):
        """ Class-aware wrapper for `::pdf_button_field_on_state()`."""
        return _mupdf.PdfObj_pdf_button_field_on_state(self)

    def pdf_choice_field_option(self, exportval, i):
        """ Class-aware wrapper for `::pdf_choice_field_option()`."""
        return _mupdf.PdfObj_pdf_choice_field_option(self, exportval, i)

    def pdf_choice_field_option_count(self):
        """ Class-aware wrapper for `::pdf_choice_field_option_count()`."""
        return _mupdf.PdfObj_pdf_choice_field_option_count(self)

    def pdf_clean_obj(self):
        """ Class-aware wrapper for `::pdf_clean_obj()`."""
        return _mupdf.PdfObj_pdf_clean_obj(self)

    def pdf_copy_array(self):
        """ Class-aware wrapper for `::pdf_copy_array()`."""
        return _mupdf.PdfObj_pdf_copy_array(self)

    def pdf_copy_dict(self):
        """ Class-aware wrapper for `::pdf_copy_dict()`."""
        return _mupdf.PdfObj_pdf_copy_dict(self)

    def pdf_debug_obj(self):
        """ Class-aware wrapper for `::pdf_debug_obj()`."""
        return _mupdf.PdfObj_pdf_debug_obj(self)

    def pdf_debug_ref(self):
        """ Class-aware wrapper for `::pdf_debug_ref()`."""
        return _mupdf.PdfObj_pdf_debug_ref(self)

    def pdf_deep_copy_obj(self):
        """ Class-aware wrapper for `::pdf_deep_copy_obj()`."""
        return _mupdf.PdfObj_pdf_deep_copy_obj(self)

    def pdf_dict_del(self, key):
        """ Class-aware wrapper for `::pdf_dict_del()`."""
        return _mupdf.PdfObj_pdf_dict_del(self, key)

    def pdf_dict_dels(self, key):
        """ Class-aware wrapper for `::pdf_dict_dels()`."""
        return _mupdf.PdfObj_pdf_dict_dels(self, key)

    def pdf_dict_get_bool(self, key):
        """ Class-aware wrapper for `::pdf_dict_get_bool()`."""
        return _mupdf.PdfObj_pdf_dict_get_bool(self, key)

    def pdf_dict_get_bool_default(self, key, _def):
        """ Class-aware wrapper for `::pdf_dict_get_bool_default()`."""
        return _mupdf.PdfObj_pdf_dict_get_bool_default(self, key, _def)

    def pdf_dict_get_date(self, key):
        """ Class-aware wrapper for `::pdf_dict_get_date()`."""
        return _mupdf.PdfObj_pdf_dict_get_date(self, key)

    def pdf_dict_get_inheritable(self, key):
        """ Class-aware wrapper for `::pdf_dict_get_inheritable()`."""
        return _mupdf.PdfObj_pdf_dict_get_inheritable(self, key)

    def pdf_dict_get_inheritable_bool(self, key):
        """ Class-aware wrapper for `::pdf_dict_get_inheritable_bool()`."""
        return _mupdf.PdfObj_pdf_dict_get_inheritable_bool(self, key)

    def pdf_dict_get_inheritable_date(self, key):
        """ Class-aware wrapper for `::pdf_dict_get_inheritable_date()`."""
        return _mupdf.PdfObj_pdf_dict_get_inheritable_date(self, key)

    def pdf_dict_get_inheritable_int(self, key):
        """ Class-aware wrapper for `::pdf_dict_get_inheritable_int()`."""
        return _mupdf.PdfObj_pdf_dict_get_inheritable_int(self, key)

    def pdf_dict_get_inheritable_int64(self, key):
        """ Class-aware wrapper for `::pdf_dict_get_inheritable_int64()`."""
        return _mupdf.PdfObj_pdf_dict_get_inheritable_int64(self, key)

    def pdf_dict_get_inheritable_matrix(self, key):
        """ Class-aware wrapper for `::pdf_dict_get_inheritable_matrix()`."""
        return _mupdf.PdfObj_pdf_dict_get_inheritable_matrix(self, key)

    def pdf_dict_get_inheritable_name(self, key):
        """ Class-aware wrapper for `::pdf_dict_get_inheritable_name()`."""
        return _mupdf.PdfObj_pdf_dict_get_inheritable_name(self, key)

    def pdf_dict_get_inheritable_real(self, key):
        """ Class-aware wrapper for `::pdf_dict_get_inheritable_real()`."""
        return _mupdf.PdfObj_pdf_dict_get_inheritable_real(self, key)

    def pdf_dict_get_inheritable_rect(self, key):
        """ Class-aware wrapper for `::pdf_dict_get_inheritable_rect()`."""
        return _mupdf.PdfObj_pdf_dict_get_inheritable_rect(self, key)

    def pdf_dict_get_inheritable_string(self, key, sizep):
        """
        Class-aware wrapper for `::pdf_dict_get_inheritable_string()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_dict_get_inheritable_string(::pdf_obj *key)` => `(const char *, size_t sizep)`
        """
        return _mupdf.PdfObj_pdf_dict_get_inheritable_string(self, key, sizep)

    def pdf_dict_get_inheritable_text_string(self, key):
        """ Class-aware wrapper for `::pdf_dict_get_inheritable_text_string()`."""
        return _mupdf.PdfObj_pdf_dict_get_inheritable_text_string(self, key)

    def pdf_dict_get_int(self, key):
        """ Class-aware wrapper for `::pdf_dict_get_int()`."""
        return _mupdf.PdfObj_pdf_dict_get_int(self, key)

    def pdf_dict_get_int64(self, key):
        """ Class-aware wrapper for `::pdf_dict_get_int64()`."""
        return _mupdf.PdfObj_pdf_dict_get_int64(self, key)

    def pdf_dict_get_int_default(self, key, _def):
        """ Class-aware wrapper for `::pdf_dict_get_int_default()`."""
        return _mupdf.PdfObj_pdf_dict_get_int_default(self, key, _def)

    def pdf_dict_get_key(self, idx):
        """ Class-aware wrapper for `::pdf_dict_get_key()`."""
        return _mupdf.PdfObj_pdf_dict_get_key(self, idx)

    def pdf_dict_get_matrix(self, key):
        """ Class-aware wrapper for `::pdf_dict_get_matrix()`."""
        return _mupdf.PdfObj_pdf_dict_get_matrix(self, key)

    def pdf_dict_get_name(self, key):
        """ Class-aware wrapper for `::pdf_dict_get_name()`."""
        return _mupdf.PdfObj_pdf_dict_get_name(self, key)

    def pdf_dict_get_real(self, key):
        """ Class-aware wrapper for `::pdf_dict_get_real()`."""
        return _mupdf.PdfObj_pdf_dict_get_real(self, key)

    def pdf_dict_get_real_default(self, key, _def):
        """ Class-aware wrapper for `::pdf_dict_get_real_default()`."""
        return _mupdf.PdfObj_pdf_dict_get_real_default(self, key, _def)

    def pdf_dict_get_rect(self, key):
        """ Class-aware wrapper for `::pdf_dict_get_rect()`."""
        return _mupdf.PdfObj_pdf_dict_get_rect(self, key)

    def pdf_dict_get_string(self, key, sizep):
        """
        Class-aware wrapper for `::pdf_dict_get_string()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_dict_get_string(::pdf_obj *key)` => `(const char *, size_t sizep)`
        """
        return _mupdf.PdfObj_pdf_dict_get_string(self, key, sizep)

    def pdf_dict_get_text_string(self, key):
        """ Class-aware wrapper for `::pdf_dict_get_text_string()`."""
        return _mupdf.PdfObj_pdf_dict_get_text_string(self, key)

    def pdf_dict_get_val(self, idx):
        """ Class-aware wrapper for `::pdf_dict_get_val()`."""
        return _mupdf.PdfObj_pdf_dict_get_val(self, idx)

    def pdf_dict_geta(self, key, abbrev):
        """ Class-aware wrapper for `::pdf_dict_geta()`."""
        return _mupdf.PdfObj_pdf_dict_geta(self, key, abbrev)

    def pdf_dict_getp(self, path):
        """ Class-aware wrapper for `::pdf_dict_getp()`."""
        return _mupdf.PdfObj_pdf_dict_getp(self, path)

    def pdf_dict_getp_inheritable(self, path):
        """ Class-aware wrapper for `::pdf_dict_getp_inheritable()`."""
        return _mupdf.PdfObj_pdf_dict_getp_inheritable(self, path)

    def pdf_dict_gets(self, key):
        """ Class-aware wrapper for `::pdf_dict_gets()`."""
        return _mupdf.PdfObj_pdf_dict_gets(self, key)

    def pdf_dict_gets_inheritable(self, key):
        """ Class-aware wrapper for `::pdf_dict_gets_inheritable()`."""
        return _mupdf.PdfObj_pdf_dict_gets_inheritable(self, key)

    def pdf_dict_getsa(self, key, abbrev):
        """ Class-aware wrapper for `::pdf_dict_getsa()`."""
        return _mupdf.PdfObj_pdf_dict_getsa(self, key, abbrev)

    def pdf_dict_len(self):
        """ Class-aware wrapper for `::pdf_dict_len()`."""
        return _mupdf.PdfObj_pdf_dict_len(self)

    def pdf_dict_put(self, key, val):
        """ Class-aware wrapper for `::pdf_dict_put()`."""
        return _mupdf.PdfObj_pdf_dict_put(self, key, val)

    def pdf_dict_put_array(self, key, initial):
        """ Class-aware wrapper for `::pdf_dict_put_array()`."""
        return _mupdf.PdfObj_pdf_dict_put_array(self, key, initial)

    def pdf_dict_put_bool(self, key, x):
        """ Class-aware wrapper for `::pdf_dict_put_bool()`."""
        return _mupdf.PdfObj_pdf_dict_put_bool(self, key, x)

    def pdf_dict_put_date(self, key, time):
        """ Class-aware wrapper for `::pdf_dict_put_date()`."""
        return _mupdf.PdfObj_pdf_dict_put_date(self, key, time)

    def pdf_dict_put_dict(self, key, initial):
        """ Class-aware wrapper for `::pdf_dict_put_dict()`."""
        return _mupdf.PdfObj_pdf_dict_put_dict(self, key, initial)

    def pdf_dict_put_int(self, key, x):
        """ Class-aware wrapper for `::pdf_dict_put_int()`."""
        return _mupdf.PdfObj_pdf_dict_put_int(self, key, x)

    def pdf_dict_put_matrix(self, key, x):
        """ Class-aware wrapper for `::pdf_dict_put_matrix()`."""
        return _mupdf.PdfObj_pdf_dict_put_matrix(self, key, x)

    def pdf_dict_put_name(self, key, x):
        """ Class-aware wrapper for `::pdf_dict_put_name()`."""
        return _mupdf.PdfObj_pdf_dict_put_name(self, key, x)

    def pdf_dict_put_real(self, key, x):
        """ Class-aware wrapper for `::pdf_dict_put_real()`."""
        return _mupdf.PdfObj_pdf_dict_put_real(self, key, x)

    def pdf_dict_put_rect(self, key, x):
        """ Class-aware wrapper for `::pdf_dict_put_rect()`."""
        return _mupdf.PdfObj_pdf_dict_put_rect(self, key, x)

    def pdf_dict_put_string(self, key, x, n):
        """ Class-aware wrapper for `::pdf_dict_put_string()`."""
        return _mupdf.PdfObj_pdf_dict_put_string(self, key, x, n)

    def pdf_dict_put_text_string(self, key, x):
        """ Class-aware wrapper for `::pdf_dict_put_text_string()`."""
        return _mupdf.PdfObj_pdf_dict_put_text_string(self, key, x)

    def pdf_dict_put_val_null(self, idx):
        """ Class-aware wrapper for `::pdf_dict_put_val_null()`."""
        return _mupdf.PdfObj_pdf_dict_put_val_null(self, idx)

    def pdf_dict_putp(self, path, val):
        """ Class-aware wrapper for `::pdf_dict_putp()`."""
        return _mupdf.PdfObj_pdf_dict_putp(self, path, val)

    def pdf_dict_puts(self, key, val):
        """ Class-aware wrapper for `::pdf_dict_puts()`."""
        return _mupdf.PdfObj_pdf_dict_puts(self, key, val)

    def pdf_dict_puts_dict(self, key, initial):
        """ Class-aware wrapper for `::pdf_dict_puts_dict()`."""
        return _mupdf.PdfObj_pdf_dict_puts_dict(self, key, initial)

    def pdf_dirty_obj(self):
        """ Class-aware wrapper for `::pdf_dirty_obj()`."""
        return _mupdf.PdfObj_pdf_dirty_obj(self)

    def pdf_field_border_style(self):
        """ Class-aware wrapper for `::pdf_field_border_style()`."""
        return _mupdf.PdfObj_pdf_field_border_style(self)

    def pdf_field_display(self):
        """ Class-aware wrapper for `::pdf_field_display()`."""
        return _mupdf.PdfObj_pdf_field_display(self)

    def pdf_field_flags(self):
        """ Class-aware wrapper for `::pdf_field_flags()`."""
        return _mupdf.PdfObj_pdf_field_flags(self)

    def pdf_field_label(self):
        """ Class-aware wrapper for `::pdf_field_label()`."""
        return _mupdf.PdfObj_pdf_field_label(self)

    def pdf_field_set_border_style(self, text):
        """ Class-aware wrapper for `::pdf_field_set_border_style()`."""
        return _mupdf.PdfObj_pdf_field_set_border_style(self, text)

    def pdf_field_set_button_caption(self, text):
        """ Class-aware wrapper for `::pdf_field_set_button_caption()`."""
        return _mupdf.PdfObj_pdf_field_set_button_caption(self, text)

    def pdf_field_set_display(self, d):
        """ Class-aware wrapper for `::pdf_field_set_display()`."""
        return _mupdf.PdfObj_pdf_field_set_display(self, d)

    def pdf_field_set_fill_color(self, col):
        """ Class-aware wrapper for `::pdf_field_set_fill_color()`."""
        return _mupdf.PdfObj_pdf_field_set_fill_color(self, col)

    def pdf_field_set_text_color(self, col):
        """ Class-aware wrapper for `::pdf_field_set_text_color()`."""
        return _mupdf.PdfObj_pdf_field_set_text_color(self, col)

    def pdf_field_type(self):
        """ Class-aware wrapper for `::pdf_field_type()`."""
        return _mupdf.PdfObj_pdf_field_type(self)

    def pdf_field_type_string(self):
        """ Class-aware wrapper for `::pdf_field_type_string()`."""
        return _mupdf.PdfObj_pdf_field_type_string(self)

    def pdf_field_value(self):
        """ Class-aware wrapper for `::pdf_field_value()`."""
        return _mupdf.PdfObj_pdf_field_value(self)

    def pdf_filter_xobject_instance(self, page_res, ctm, options, cycle_up):
        """ Class-aware wrapper for `::pdf_filter_xobject_instance()`."""
        return _mupdf.PdfObj_pdf_filter_xobject_instance(self, page_res, ctm, options, cycle_up)

    def pdf_flatten_inheritable_page_items(self):
        """ Class-aware wrapper for `::pdf_flatten_inheritable_page_items()`."""
        return _mupdf.PdfObj_pdf_flatten_inheritable_page_items(self)

    def pdf_get_bound_document(self):
        """ Class-aware wrapper for `::pdf_get_bound_document()`."""
        return _mupdf.PdfObj_pdf_get_bound_document(self)

    def pdf_get_embedded_file_params(self, out):
        """ Class-aware wrapper for `::pdf_get_embedded_file_params()`."""
        return _mupdf.PdfObj_pdf_get_embedded_file_params(self, out)

    def pdf_get_indirect_document(self):
        """ Class-aware wrapper for `::pdf_get_indirect_document()`."""
        return _mupdf.PdfObj_pdf_get_indirect_document(self)

    def pdf_is_array(self):
        """ Class-aware wrapper for `::pdf_is_array()`."""
        return _mupdf.PdfObj_pdf_is_array(self)

    def pdf_is_bool(self):
        """ Class-aware wrapper for `::pdf_is_bool()`."""
        return _mupdf.PdfObj_pdf_is_bool(self)

    def pdf_is_dict(self):
        """ Class-aware wrapper for `::pdf_is_dict()`."""
        return _mupdf.PdfObj_pdf_is_dict(self)

    def pdf_is_embedded_file(self):
        """ Class-aware wrapper for `::pdf_is_embedded_file()`."""
        return _mupdf.PdfObj_pdf_is_embedded_file(self)

    def pdf_is_indirect(self):
        """ Class-aware wrapper for `::pdf_is_indirect()`."""
        return _mupdf.PdfObj_pdf_is_indirect(self)

    def pdf_is_int(self):
        """ Class-aware wrapper for `::pdf_is_int()`."""
        return _mupdf.PdfObj_pdf_is_int(self)

    def pdf_is_jpx_image(self):
        """ Class-aware wrapper for `::pdf_is_jpx_image()`."""
        return _mupdf.PdfObj_pdf_is_jpx_image(self)

    def pdf_is_name(self):
        """ Class-aware wrapper for `::pdf_is_name()`."""
        return _mupdf.PdfObj_pdf_is_name(self)

    def pdf_is_null(self):
        """ Class-aware wrapper for `::pdf_is_null()`."""
        return _mupdf.PdfObj_pdf_is_null(self)

    def pdf_is_number(self):
        """ Class-aware wrapper for `::pdf_is_number()`."""
        return _mupdf.PdfObj_pdf_is_number(self)

    def pdf_is_real(self):
        """ Class-aware wrapper for `::pdf_is_real()`."""
        return _mupdf.PdfObj_pdf_is_real(self)

    def pdf_is_stream(self):
        """ Class-aware wrapper for `::pdf_is_stream()`."""
        return _mupdf.PdfObj_pdf_is_stream(self)

    def pdf_is_string(self):
        """ Class-aware wrapper for `::pdf_is_string()`."""
        return _mupdf.PdfObj_pdf_is_string(self)

    def pdf_line_ending_from_name(self):
        """ Class-aware wrapper for `::pdf_line_ending_from_name()`."""
        return _mupdf.PdfObj_pdf_line_ending_from_name(self)

    def pdf_load_colorspace(self):
        """ Class-aware wrapper for `::pdf_load_colorspace()`."""
        return _mupdf.PdfObj_pdf_load_colorspace(self)

    def pdf_load_embedded_file_contents(self):
        """ Class-aware wrapper for `::pdf_load_embedded_file_contents()`."""
        return _mupdf.PdfObj_pdf_load_embedded_file_contents(self)

    def pdf_load_field_name(self):
        """ Class-aware wrapper for `::pdf_load_field_name()`."""
        return _mupdf.PdfObj_pdf_load_field_name(self)

    def pdf_load_function(self, _in, out):
        """ Class-aware wrapper for `::pdf_load_function()`."""
        return _mupdf.PdfObj_pdf_load_function(self, _in, out)

    def pdf_load_raw_stream(self):
        """ Class-aware wrapper for `::pdf_load_raw_stream()`."""
        return _mupdf.PdfObj_pdf_load_raw_stream(self)

    def pdf_load_stream(self):
        """ Class-aware wrapper for `::pdf_load_stream()`."""
        return _mupdf.PdfObj_pdf_load_stream(self)

    def pdf_load_stream_or_string_as_utf8(self):
        """ Class-aware wrapper for `::pdf_load_stream_or_string_as_utf8()`."""
        return _mupdf.PdfObj_pdf_load_stream_or_string_as_utf8(self)

    def pdf_lookup_field(self, name):
        """ Class-aware wrapper for `::pdf_lookup_field()`."""
        return _mupdf.PdfObj_pdf_lookup_field(self, name)

    def pdf_lookup_number(self, needle):
        """ Class-aware wrapper for `::pdf_lookup_number()`."""
        return _mupdf.PdfObj_pdf_lookup_number(self, needle)

    def pdf_mark_obj(self):
        """ Class-aware wrapper for `::pdf_mark_obj()`."""
        return _mupdf.PdfObj_pdf_mark_obj(self)

    def pdf_name_eq(self, b):
        """ Class-aware wrapper for `::pdf_name_eq()`."""
        return _mupdf.PdfObj_pdf_name_eq(self, b)

    def pdf_new_utf8_from_pdf_stream_obj(self):
        """ Class-aware wrapper for `::pdf_new_utf8_from_pdf_stream_obj()`."""
        return _mupdf.PdfObj_pdf_new_utf8_from_pdf_stream_obj(self)

    def pdf_new_utf8_from_pdf_string_obj(self):
        """ Class-aware wrapper for `::pdf_new_utf8_from_pdf_string_obj()`."""
        return _mupdf.PdfObj_pdf_new_utf8_from_pdf_string_obj(self)

    def pdf_obj_is_dirty(self):
        """ Class-aware wrapper for `::pdf_obj_is_dirty()`."""
        return _mupdf.PdfObj_pdf_obj_is_dirty(self)

    def pdf_obj_is_incremental(self):
        """ Class-aware wrapper for `::pdf_obj_is_incremental()`."""
        return _mupdf.PdfObj_pdf_obj_is_incremental(self)

    def pdf_obj_marked(self):
        """ Class-aware wrapper for `::pdf_obj_marked()`."""
        return _mupdf.PdfObj_pdf_obj_marked(self)

    def pdf_obj_memo(self, bit, memo):
        """
        Class-aware wrapper for `::pdf_obj_memo()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_obj_memo(int bit)` => `(int, int memo)`
        """
        return _mupdf.PdfObj_pdf_obj_memo(self, bit, memo)

    def pdf_obj_parent_num(self):
        """ Class-aware wrapper for `::pdf_obj_parent_num()`."""
        return _mupdf.PdfObj_pdf_obj_parent_num(self)

    def pdf_obj_refs(self):
        """ Class-aware wrapper for `::pdf_obj_refs()`."""
        return _mupdf.PdfObj_pdf_obj_refs(self)

    def pdf_objcmp(self, b):
        """ Class-aware wrapper for `::pdf_objcmp()`."""
        return _mupdf.PdfObj_pdf_objcmp(self, b)

    def pdf_objcmp_deep(self, b):
        """ Class-aware wrapper for `::pdf_objcmp_deep()`."""
        return _mupdf.PdfObj_pdf_objcmp_deep(self, b)

    def pdf_objcmp_resolve(self, b):
        """ Class-aware wrapper for `::pdf_objcmp_resolve()`."""
        return _mupdf.PdfObj_pdf_objcmp_resolve(self, b)

    def pdf_open_raw_stream(self):
        """ Class-aware wrapper for `::pdf_open_raw_stream()`."""
        return _mupdf.PdfObj_pdf_open_raw_stream(self)

    def pdf_open_stream(self):
        """ Class-aware wrapper for `::pdf_open_stream()`."""
        return _mupdf.PdfObj_pdf_open_stream(self)

    def pdf_page_obj_transform(self, outbox, outctm):
        """ Class-aware wrapper for `::pdf_page_obj_transform()`."""
        return _mupdf.PdfObj_pdf_page_obj_transform(self, outbox, outctm)

    def pdf_page_obj_transform_box(self, outbox, out, box):
        """ Class-aware wrapper for `::pdf_page_obj_transform_box()`."""
        return _mupdf.PdfObj_pdf_page_obj_transform_box(self, outbox, out, box)

    def pdf_pin_document(self):
        """ Class-aware wrapper for `::pdf_pin_document()`."""
        return _mupdf.PdfObj_pdf_pin_document(self)

    def pdf_recolor_shade(self, reshade, opaque):
        """
        Class-aware wrapper for `::pdf_recolor_shade()`.
        	Recolor a shade.
        """
        return _mupdf.PdfObj_pdf_recolor_shade(self, reshade, opaque)

    def pdf_resolve_indirect(self):
        """
        Class-aware wrapper for `::pdf_resolve_indirect()`.
        	Resolve an indirect object (or chain of objects).

        	This can cause xref reorganisations (solidifications etc) due to
        	repairs, so all held pdf_xref_entries should be considered
        	invalid after this call (other than the returned one).
        """
        return _mupdf.PdfObj_pdf_resolve_indirect(self)

    def pdf_resolve_indirect_chain(self):
        """ Class-aware wrapper for `::pdf_resolve_indirect_chain()`."""
        return _mupdf.PdfObj_pdf_resolve_indirect_chain(self)

    def pdf_set_int(self, i):
        """ Class-aware wrapper for `::pdf_set_int()`."""
        return _mupdf.PdfObj_pdf_set_int(self, i)

    def pdf_set_obj_memo(self, bit, memo):
        """ Class-aware wrapper for `::pdf_set_obj_memo()`."""
        return _mupdf.PdfObj_pdf_set_obj_memo(self, bit, memo)

    def pdf_set_obj_parent(self, num):
        """ Class-aware wrapper for `::pdf_set_obj_parent()`."""
        return _mupdf.PdfObj_pdf_set_obj_parent(self, num)

    def pdf_set_str_len(self, newlen):
        """ Class-aware wrapper for `::pdf_set_str_len()`."""
        return _mupdf.PdfObj_pdf_set_str_len(self, newlen)

    def pdf_sort_dict(self):
        """ Class-aware wrapper for `::pdf_sort_dict()`."""
        return _mupdf.PdfObj_pdf_sort_dict(self)

    def pdf_store_item(self, val, itemsize):
        """ Class-aware wrapper for `::pdf_store_item()`."""
        return _mupdf.PdfObj_pdf_store_item(self, val, itemsize)

    def pdf_to_bool(self):
        """ Class-aware wrapper for `::pdf_to_bool()`."""
        return _mupdf.PdfObj_pdf_to_bool(self)

    def pdf_to_bool_default(self, _def):
        """ Class-aware wrapper for `::pdf_to_bool_default()`."""
        return _mupdf.PdfObj_pdf_to_bool_default(self, _def)

    def pdf_to_date(self):
        """ Class-aware wrapper for `::pdf_to_date()`."""
        return _mupdf.PdfObj_pdf_to_date(self)

    def pdf_to_gen(self):
        """ Class-aware wrapper for `::pdf_to_gen()`."""
        return _mupdf.PdfObj_pdf_to_gen(self)

    def pdf_to_int(self):
        """ Class-aware wrapper for `::pdf_to_int()`."""
        return _mupdf.PdfObj_pdf_to_int(self)

    def pdf_to_int64(self):
        """ Class-aware wrapper for `::pdf_to_int64()`."""
        return _mupdf.PdfObj_pdf_to_int64(self)

    def pdf_to_int_default(self, _def):
        """ Class-aware wrapper for `::pdf_to_int_default()`."""
        return _mupdf.PdfObj_pdf_to_int_default(self, _def)

    def pdf_to_matrix(self):
        """ Class-aware wrapper for `::pdf_to_matrix()`."""
        return _mupdf.PdfObj_pdf_to_matrix(self)

    def pdf_to_name(self):
        """ Class-aware wrapper for `::pdf_to_name()`."""
        return _mupdf.PdfObj_pdf_to_name(self)

    def pdf_to_num(self):
        """ Class-aware wrapper for `::pdf_to_num()`."""
        return _mupdf.PdfObj_pdf_to_num(self)

    def pdf_to_quad(self, offset):
        """ Class-aware wrapper for `::pdf_to_quad()`."""
        return _mupdf.PdfObj_pdf_to_quad(self, offset)

    def pdf_to_real(self):
        """ Class-aware wrapper for `::pdf_to_real()`."""
        return _mupdf.PdfObj_pdf_to_real(self)

    def pdf_to_real_default(self, _def):
        """ Class-aware wrapper for `::pdf_to_real_default()`."""
        return _mupdf.PdfObj_pdf_to_real_default(self, _def)

    def pdf_to_rect(self):
        """ Class-aware wrapper for `::pdf_to_rect()`."""
        return _mupdf.PdfObj_pdf_to_rect(self)

    def pdf_to_str_buf(self):
        """ Class-aware wrapper for `::pdf_to_str_buf()`."""
        return _mupdf.PdfObj_pdf_to_str_buf(self)

    def pdf_to_str_len(self):
        """ Class-aware wrapper for `::pdf_to_str_len()`."""
        return _mupdf.PdfObj_pdf_to_str_len(self)

    def pdf_to_string(self, sizep):
        """
        Class-aware wrapper for `::pdf_to_string()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_to_string()` => `(const char *, size_t sizep)`
        """
        return _mupdf.PdfObj_pdf_to_string(self, sizep)

    def pdf_to_text_string(self):
        """ Class-aware wrapper for `::pdf_to_text_string()`."""
        return _mupdf.PdfObj_pdf_to_text_string(self)

    def pdf_unmark_obj(self):
        """ Class-aware wrapper for `::pdf_unmark_obj()`."""
        return _mupdf.PdfObj_pdf_unmark_obj(self)

    def pdf_verify_embedded_file_checksum(self):
        """ Class-aware wrapper for `::pdf_verify_embedded_file_checksum()`."""
        return _mupdf.PdfObj_pdf_verify_embedded_file_checksum(self)

    def pdf_walk_tree(self, kid_name, arrive, leave, arg, names, values):
        """
        Class-aware wrapper for `::pdf_walk_tree()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_walk_tree(::pdf_obj *kid_name, void (*arrive)(::fz_context *, ::pdf_obj *, void *, ::pdf_obj **), void (*leave)(::fz_context *, ::pdf_obj *, void *), void *arg, ::pdf_obj **names, ::pdf_obj **values)` => `()`
        """
        return _mupdf.PdfObj_pdf_walk_tree(self, kid_name, arrive, leave, arg, names, values)

    def pdf_xobject_bbox(self):
        """ Class-aware wrapper for `::pdf_xobject_bbox()`."""
        return _mupdf.PdfObj_pdf_xobject_bbox(self)

    def pdf_xobject_colorspace(self):
        """ Class-aware wrapper for `::pdf_xobject_colorspace()`."""
        return _mupdf.PdfObj_pdf_xobject_colorspace(self)

    def pdf_xobject_isolated(self):
        """ Class-aware wrapper for `::pdf_xobject_isolated()`."""
        return _mupdf.PdfObj_pdf_xobject_isolated(self)

    def pdf_xobject_knockout(self):
        """ Class-aware wrapper for `::pdf_xobject_knockout()`."""
        return _mupdf.PdfObj_pdf_xobject_knockout(self)

    def pdf_xobject_matrix(self):
        """ Class-aware wrapper for `::pdf_xobject_matrix()`."""
        return _mupdf.PdfObj_pdf_xobject_matrix(self)

    def pdf_xobject_resources(self):
        """ Class-aware wrapper for `::pdf_xobject_resources()`."""
        return _mupdf.PdfObj_pdf_xobject_resources(self)

    def pdf_xobject_transparency(self):
        """ Class-aware wrapper for `::pdf_xobject_transparency()`."""
        return _mupdf.PdfObj_pdf_xobject_transparency(self)

    def pdf_dict_get(self, *args):
        """
        *Overload 1:*
        Class-aware wrapper for `::pdf_dict_get()`.

        |

        *Overload 2:*
        Typesafe wrapper for looking up things such as PDF_ENUM_NAME_Annots.
        """
        return _mupdf.PdfObj_pdf_dict_get(self, *args)

    def pdf_load_field_name2(self):
        """ Alternative to `pdf_load_field_name()` that returns a std::string."""
        return _mupdf.PdfObj_pdf_load_field_name2(self)

    def __init__(self, *args):
        """
        *Overload 1:*
        == Constructors.  Constructor using `pdf_new_action_from_link()`.

        |

        *Overload 2:*
        Constructor using `pdf_new_array()`.

        |

        *Overload 3:*
        Constructor using `pdf_new_date()`.

        |

        *Overload 4:*
        Constructor using `pdf_new_dest_from_link()`.

        |

        *Overload 5:*
        Constructor using `pdf_new_indirect()`.

        |

        *Overload 6:*
        Constructor using `pdf_new_int()`.

        |

        *Overload 7:*
        Constructor using `pdf_new_matrix()`.

        |

        *Overload 8:*
        Constructor using `pdf_new_name()`.

        |

        *Overload 9:*
        Constructor using `pdf_new_real()`.

        |

        *Overload 10:*
        Constructor using `pdf_new_rect()`.

        |

        *Overload 11:*
        Constructor using `pdf_new_string()`.

        |

        *Overload 12:*
        Constructor using `pdf_new_xobject()`.

        |

        *Overload 13:*
        Copy constructor using `pdf_keep_obj()`.

        |

        *Overload 14:*
        Constructor using raw copy of pre-existing `::pdf_obj`.

        |

        *Overload 15:*
        Constructor using raw copy of pre-existing `::pdf_obj`.
        """
        _mupdf.PdfObj_swiginit(self, _mupdf.new_PdfObj(*args))
    __swig_destroy__ = _mupdf.delete_PdfObj

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.PdfObj_m_internal_value(self)
    m_internal = property(_mupdf.PdfObj_m_internal_get, _mupdf.PdfObj_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.PdfObj_s_num_instances_get, _mupdf.PdfObj_s_num_instances_set)