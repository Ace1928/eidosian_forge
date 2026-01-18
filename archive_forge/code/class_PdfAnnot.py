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
class PdfAnnot(object):
    """ Wrapper class for struct `pdf_annot`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def pdf_add_annot_border_dash_item(self, length):
        """ Class-aware wrapper for `::pdf_add_annot_border_dash_item()`."""
        return _mupdf.PdfAnnot_pdf_add_annot_border_dash_item(self, length)

    def pdf_add_annot_ink_list_stroke(self):
        """ Class-aware wrapper for `::pdf_add_annot_ink_list_stroke()`."""
        return _mupdf.PdfAnnot_pdf_add_annot_ink_list_stroke(self)

    def pdf_add_annot_ink_list_stroke_vertex(self, p):
        """ Class-aware wrapper for `::pdf_add_annot_ink_list_stroke_vertex()`."""
        return _mupdf.PdfAnnot_pdf_add_annot_ink_list_stroke_vertex(self, p)

    def pdf_add_annot_quad_point(self, quad):
        """ Class-aware wrapper for `::pdf_add_annot_quad_point()`."""
        return _mupdf.PdfAnnot_pdf_add_annot_quad_point(self, quad)

    def pdf_add_annot_vertex(self, p):
        """ Class-aware wrapper for `::pdf_add_annot_vertex()`."""
        return _mupdf.PdfAnnot_pdf_add_annot_vertex(self, p)

    def pdf_annot_MK_BC(self, n, color):
        """
        Class-aware wrapper for `::pdf_annot_MK_BC()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_annot_MK_BC(float color[4])` => int n
        """
        return _mupdf.PdfAnnot_pdf_annot_MK_BC(self, n, color)

    def pdf_annot_MK_BC_rgb(self, rgb):
        """ Class-aware wrapper for `::pdf_annot_MK_BC_rgb()`."""
        return _mupdf.PdfAnnot_pdf_annot_MK_BC_rgb(self, rgb)

    def pdf_annot_MK_BG(self, n, color):
        """
        Class-aware wrapper for `::pdf_annot_MK_BG()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_annot_MK_BG(float color[4])` => int n
        """
        return _mupdf.PdfAnnot_pdf_annot_MK_BG(self, n, color)

    def pdf_annot_MK_BG_rgb(self, rgb):
        """ Class-aware wrapper for `::pdf_annot_MK_BG_rgb()`."""
        return _mupdf.PdfAnnot_pdf_annot_MK_BG_rgb(self, rgb)

    def pdf_annot_active(self):
        """ Class-aware wrapper for `::pdf_annot_active()`."""
        return _mupdf.PdfAnnot_pdf_annot_active(self)

    def pdf_annot_ap(self):
        """ Class-aware wrapper for `::pdf_annot_ap()`."""
        return _mupdf.PdfAnnot_pdf_annot_ap(self)

    def pdf_annot_author(self):
        """ Class-aware wrapper for `::pdf_annot_author()`."""
        return _mupdf.PdfAnnot_pdf_annot_author(self)

    def pdf_annot_border(self):
        """ Class-aware wrapper for `::pdf_annot_border()`."""
        return _mupdf.PdfAnnot_pdf_annot_border(self)

    def pdf_annot_border_dash_count(self):
        """ Class-aware wrapper for `::pdf_annot_border_dash_count()`."""
        return _mupdf.PdfAnnot_pdf_annot_border_dash_count(self)

    def pdf_annot_border_dash_item(self, i):
        """ Class-aware wrapper for `::pdf_annot_border_dash_item()`."""
        return _mupdf.PdfAnnot_pdf_annot_border_dash_item(self, i)

    def pdf_annot_border_effect(self):
        """ Class-aware wrapper for `::pdf_annot_border_effect()`."""
        return _mupdf.PdfAnnot_pdf_annot_border_effect(self)

    def pdf_annot_border_effect_intensity(self):
        """ Class-aware wrapper for `::pdf_annot_border_effect_intensity()`."""
        return _mupdf.PdfAnnot_pdf_annot_border_effect_intensity(self)

    def pdf_annot_border_style(self):
        """ Class-aware wrapper for `::pdf_annot_border_style()`."""
        return _mupdf.PdfAnnot_pdf_annot_border_style(self)

    def pdf_annot_border_width(self):
        """ Class-aware wrapper for `::pdf_annot_border_width()`."""
        return _mupdf.PdfAnnot_pdf_annot_border_width(self)

    def pdf_annot_color(self, n, color):
        """
        Class-aware wrapper for `::pdf_annot_color()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_annot_color(float color[4])` => int n
        """
        return _mupdf.PdfAnnot_pdf_annot_color(self, n, color)

    def pdf_annot_contents(self):
        """ Class-aware wrapper for `::pdf_annot_contents()`."""
        return _mupdf.PdfAnnot_pdf_annot_contents(self)

    def pdf_annot_creation_date(self):
        """ Class-aware wrapper for `::pdf_annot_creation_date()`."""
        return _mupdf.PdfAnnot_pdf_annot_creation_date(self)

    def pdf_annot_default_appearance(self, font, size, n, color):
        """
        Class-aware wrapper for `::pdf_annot_default_appearance()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_annot_default_appearance(float color[4])` => `(const char *font, float size, int n)`
        """
        return _mupdf.PdfAnnot_pdf_annot_default_appearance(self, font, size, n, color)

    def pdf_annot_ensure_local_xref(self):
        """ Class-aware wrapper for `::pdf_annot_ensure_local_xref()`."""
        return _mupdf.PdfAnnot_pdf_annot_ensure_local_xref(self)

    def pdf_annot_event_blur(self):
        """ Class-aware wrapper for `::pdf_annot_event_blur()`."""
        return _mupdf.PdfAnnot_pdf_annot_event_blur(self)

    def pdf_annot_event_down(self):
        """ Class-aware wrapper for `::pdf_annot_event_down()`."""
        return _mupdf.PdfAnnot_pdf_annot_event_down(self)

    def pdf_annot_event_enter(self):
        """ Class-aware wrapper for `::pdf_annot_event_enter()`."""
        return _mupdf.PdfAnnot_pdf_annot_event_enter(self)

    def pdf_annot_event_exit(self):
        """ Class-aware wrapper for `::pdf_annot_event_exit()`."""
        return _mupdf.PdfAnnot_pdf_annot_event_exit(self)

    def pdf_annot_event_focus(self):
        """ Class-aware wrapper for `::pdf_annot_event_focus()`."""
        return _mupdf.PdfAnnot_pdf_annot_event_focus(self)

    def pdf_annot_event_page_close(self):
        """ Class-aware wrapper for `::pdf_annot_event_page_close()`."""
        return _mupdf.PdfAnnot_pdf_annot_event_page_close(self)

    def pdf_annot_event_page_invisible(self):
        """ Class-aware wrapper for `::pdf_annot_event_page_invisible()`."""
        return _mupdf.PdfAnnot_pdf_annot_event_page_invisible(self)

    def pdf_annot_event_page_open(self):
        """ Class-aware wrapper for `::pdf_annot_event_page_open()`."""
        return _mupdf.PdfAnnot_pdf_annot_event_page_open(self)

    def pdf_annot_event_page_visible(self):
        """ Class-aware wrapper for `::pdf_annot_event_page_visible()`."""
        return _mupdf.PdfAnnot_pdf_annot_event_page_visible(self)

    def pdf_annot_event_up(self):
        """ Class-aware wrapper for `::pdf_annot_event_up()`."""
        return _mupdf.PdfAnnot_pdf_annot_event_up(self)

    def pdf_annot_field_flags(self):
        """ Class-aware wrapper for `::pdf_annot_field_flags()`."""
        return _mupdf.PdfAnnot_pdf_annot_field_flags(self)

    def pdf_annot_field_label(self):
        """ Class-aware wrapper for `::pdf_annot_field_label()`."""
        return _mupdf.PdfAnnot_pdf_annot_field_label(self)

    def pdf_annot_field_value(self):
        """ Class-aware wrapper for `::pdf_annot_field_value()`."""
        return _mupdf.PdfAnnot_pdf_annot_field_value(self)

    def pdf_annot_filespec(self):
        """ Class-aware wrapper for `::pdf_annot_filespec()`."""
        return _mupdf.PdfAnnot_pdf_annot_filespec(self)

    def pdf_annot_flags(self):
        """ Class-aware wrapper for `::pdf_annot_flags()`."""
        return _mupdf.PdfAnnot_pdf_annot_flags(self)

    def pdf_annot_has_author(self):
        """ Class-aware wrapper for `::pdf_annot_has_author()`."""
        return _mupdf.PdfAnnot_pdf_annot_has_author(self)

    def pdf_annot_has_border(self):
        """ Class-aware wrapper for `::pdf_annot_has_border()`."""
        return _mupdf.PdfAnnot_pdf_annot_has_border(self)

    def pdf_annot_has_border_effect(self):
        """ Class-aware wrapper for `::pdf_annot_has_border_effect()`."""
        return _mupdf.PdfAnnot_pdf_annot_has_border_effect(self)

    def pdf_annot_has_filespec(self):
        """ Class-aware wrapper for `::pdf_annot_has_filespec()`."""
        return _mupdf.PdfAnnot_pdf_annot_has_filespec(self)

    def pdf_annot_has_icon_name(self):
        """ Class-aware wrapper for `::pdf_annot_has_icon_name()`."""
        return _mupdf.PdfAnnot_pdf_annot_has_icon_name(self)

    def pdf_annot_has_ink_list(self):
        """ Class-aware wrapper for `::pdf_annot_has_ink_list()`."""
        return _mupdf.PdfAnnot_pdf_annot_has_ink_list(self)

    def pdf_annot_has_interior_color(self):
        """ Class-aware wrapper for `::pdf_annot_has_interior_color()`."""
        return _mupdf.PdfAnnot_pdf_annot_has_interior_color(self)

    def pdf_annot_has_line(self):
        """ Class-aware wrapper for `::pdf_annot_has_line()`."""
        return _mupdf.PdfAnnot_pdf_annot_has_line(self)

    def pdf_annot_has_line_ending_styles(self):
        """ Class-aware wrapper for `::pdf_annot_has_line_ending_styles()`."""
        return _mupdf.PdfAnnot_pdf_annot_has_line_ending_styles(self)

    def pdf_annot_has_open(self):
        """ Class-aware wrapper for `::pdf_annot_has_open()`."""
        return _mupdf.PdfAnnot_pdf_annot_has_open(self)

    def pdf_annot_has_quad_points(self):
        """ Class-aware wrapper for `::pdf_annot_has_quad_points()`."""
        return _mupdf.PdfAnnot_pdf_annot_has_quad_points(self)

    def pdf_annot_has_quadding(self):
        """ Class-aware wrapper for `::pdf_annot_has_quadding()`."""
        return _mupdf.PdfAnnot_pdf_annot_has_quadding(self)

    def pdf_annot_has_rect(self):
        """ Class-aware wrapper for `::pdf_annot_has_rect()`."""
        return _mupdf.PdfAnnot_pdf_annot_has_rect(self)

    def pdf_annot_has_vertices(self):
        """ Class-aware wrapper for `::pdf_annot_has_vertices()`."""
        return _mupdf.PdfAnnot_pdf_annot_has_vertices(self)

    def pdf_annot_hidden_for_editing(self):
        """ Class-aware wrapper for `::pdf_annot_hidden_for_editing()`."""
        return _mupdf.PdfAnnot_pdf_annot_hidden_for_editing(self)

    def pdf_annot_hot(self):
        """ Class-aware wrapper for `::pdf_annot_hot()`."""
        return _mupdf.PdfAnnot_pdf_annot_hot(self)

    def pdf_annot_icon_name(self):
        """ Class-aware wrapper for `::pdf_annot_icon_name()`."""
        return _mupdf.PdfAnnot_pdf_annot_icon_name(self)

    def pdf_annot_ink_list_count(self):
        """ Class-aware wrapper for `::pdf_annot_ink_list_count()`."""
        return _mupdf.PdfAnnot_pdf_annot_ink_list_count(self)

    def pdf_annot_ink_list_stroke_count(self, i):
        """ Class-aware wrapper for `::pdf_annot_ink_list_stroke_count()`."""
        return _mupdf.PdfAnnot_pdf_annot_ink_list_stroke_count(self, i)

    def pdf_annot_ink_list_stroke_vertex(self, i, k):
        """ Class-aware wrapper for `::pdf_annot_ink_list_stroke_vertex()`."""
        return _mupdf.PdfAnnot_pdf_annot_ink_list_stroke_vertex(self, i, k)

    def pdf_annot_interior_color(self, n, color):
        """
        Class-aware wrapper for `::pdf_annot_interior_color()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_annot_interior_color(float color[4])` => int n
        """
        return _mupdf.PdfAnnot_pdf_annot_interior_color(self, n, color)

    def pdf_annot_is_open(self):
        """ Class-aware wrapper for `::pdf_annot_is_open()`."""
        return _mupdf.PdfAnnot_pdf_annot_is_open(self)

    def pdf_annot_is_standard_stamp(self):
        """ Class-aware wrapper for `::pdf_annot_is_standard_stamp()`."""
        return _mupdf.PdfAnnot_pdf_annot_is_standard_stamp(self)

    def pdf_annot_line(self, a, b):
        """ Class-aware wrapper for `::pdf_annot_line()`."""
        return _mupdf.PdfAnnot_pdf_annot_line(self, a, b)

    def pdf_annot_line_end_style(self):
        """ Class-aware wrapper for `::pdf_annot_line_end_style()`."""
        return _mupdf.PdfAnnot_pdf_annot_line_end_style(self)

    def pdf_annot_line_ending_styles(self, start_style, end_style):
        """
        Class-aware wrapper for `::pdf_annot_line_ending_styles()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_annot_line_ending_styles()` => `(enum pdf_line_ending start_style, enum pdf_line_ending end_style)`
        """
        return _mupdf.PdfAnnot_pdf_annot_line_ending_styles(self, start_style, end_style)

    def pdf_annot_line_start_style(self):
        """ Class-aware wrapper for `::pdf_annot_line_start_style()`."""
        return _mupdf.PdfAnnot_pdf_annot_line_start_style(self)

    def pdf_annot_modification_date(self):
        """ Class-aware wrapper for `::pdf_annot_modification_date()`."""
        return _mupdf.PdfAnnot_pdf_annot_modification_date(self)

    def pdf_annot_needs_resynthesis(self):
        """ Class-aware wrapper for `::pdf_annot_needs_resynthesis()`."""
        return _mupdf.PdfAnnot_pdf_annot_needs_resynthesis(self)

    def pdf_annot_obj(self):
        """ Class-aware wrapper for `::pdf_annot_obj()`."""
        return _mupdf.PdfAnnot_pdf_annot_obj(self)

    def pdf_annot_opacity(self):
        """ Class-aware wrapper for `::pdf_annot_opacity()`."""
        return _mupdf.PdfAnnot_pdf_annot_opacity(self)

    def pdf_annot_page(self):
        """ Class-aware wrapper for `::pdf_annot_page()`."""
        return _mupdf.PdfAnnot_pdf_annot_page(self)

    def pdf_annot_pop_and_discard_local_xref(self):
        """ Class-aware wrapper for `::pdf_annot_pop_and_discard_local_xref()`."""
        return _mupdf.PdfAnnot_pdf_annot_pop_and_discard_local_xref(self)

    def pdf_annot_pop_local_xref(self):
        """ Class-aware wrapper for `::pdf_annot_pop_local_xref()`."""
        return _mupdf.PdfAnnot_pdf_annot_pop_local_xref(self)

    def pdf_annot_popup(self):
        """ Class-aware wrapper for `::pdf_annot_popup()`."""
        return _mupdf.PdfAnnot_pdf_annot_popup(self)

    def pdf_annot_push_local_xref(self):
        """ Class-aware wrapper for `::pdf_annot_push_local_xref()`."""
        return _mupdf.PdfAnnot_pdf_annot_push_local_xref(self)

    def pdf_annot_quad_point(self, i):
        """ Class-aware wrapper for `::pdf_annot_quad_point()`."""
        return _mupdf.PdfAnnot_pdf_annot_quad_point(self, i)

    def pdf_annot_quad_point_count(self):
        """ Class-aware wrapper for `::pdf_annot_quad_point_count()`."""
        return _mupdf.PdfAnnot_pdf_annot_quad_point_count(self)

    def pdf_annot_quadding(self):
        """ Class-aware wrapper for `::pdf_annot_quadding()`."""
        return _mupdf.PdfAnnot_pdf_annot_quadding(self)

    def pdf_annot_rect(self):
        """ Class-aware wrapper for `::pdf_annot_rect()`."""
        return _mupdf.PdfAnnot_pdf_annot_rect(self)

    def pdf_annot_request_resynthesis(self):
        """ Class-aware wrapper for `::pdf_annot_request_resynthesis()`."""
        return _mupdf.PdfAnnot_pdf_annot_request_resynthesis(self)

    def pdf_annot_request_synthesis(self):
        """ Class-aware wrapper for `::pdf_annot_request_synthesis()`."""
        return _mupdf.PdfAnnot_pdf_annot_request_synthesis(self)

    def pdf_annot_transform(self):
        """ Class-aware wrapper for `::pdf_annot_transform()`."""
        return _mupdf.PdfAnnot_pdf_annot_transform(self)

    def pdf_annot_type(self):
        """ Class-aware wrapper for `::pdf_annot_type()`."""
        return _mupdf.PdfAnnot_pdf_annot_type(self)

    def pdf_annot_vertex(self, i):
        """ Class-aware wrapper for `::pdf_annot_vertex()`."""
        return _mupdf.PdfAnnot_pdf_annot_vertex(self, i)

    def pdf_annot_vertex_count(self):
        """ Class-aware wrapper for `::pdf_annot_vertex_count()`."""
        return _mupdf.PdfAnnot_pdf_annot_vertex_count(self)

    def pdf_apply_redaction(self, opts):
        """ Class-aware wrapper for `::pdf_apply_redaction()`."""
        return _mupdf.PdfAnnot_pdf_apply_redaction(self, opts)

    def pdf_bound_annot(self):
        """ Class-aware wrapper for `::pdf_bound_annot()`."""
        return _mupdf.PdfAnnot_pdf_bound_annot(self)

    def pdf_bound_widget(self):
        """ Class-aware wrapper for `::pdf_bound_widget()`."""
        return _mupdf.PdfAnnot_pdf_bound_widget(self)

    def pdf_choice_widget_is_multiselect(self):
        """ Class-aware wrapper for `::pdf_choice_widget_is_multiselect()`."""
        return _mupdf.PdfAnnot_pdf_choice_widget_is_multiselect(self)

    def pdf_choice_widget_options(self, exportval, opts):
        """ Class-aware wrapper for `::pdf_choice_widget_options()`."""
        return _mupdf.PdfAnnot_pdf_choice_widget_options(self, exportval, opts)

    def pdf_choice_widget_options2(self, exportval):
        """
        Class-aware wrapper for `::pdf_choice_widget_options2()`.   Swig-friendly wrapper for pdf_choice_widget_options(), returns the
        options directly in a vector.
        """
        return _mupdf.PdfAnnot_pdf_choice_widget_options2(self, exportval)

    def pdf_choice_widget_set_value(self, n, opts):
        """ Class-aware wrapper for `::pdf_choice_widget_set_value()`."""
        return _mupdf.PdfAnnot_pdf_choice_widget_set_value(self, n, opts)

    def pdf_choice_widget_value(self, opts):
        """ Class-aware wrapper for `::pdf_choice_widget_value()`."""
        return _mupdf.PdfAnnot_pdf_choice_widget_value(self, opts)

    def pdf_clear_annot_border_dash(self):
        """ Class-aware wrapper for `::pdf_clear_annot_border_dash()`."""
        return _mupdf.PdfAnnot_pdf_clear_annot_border_dash(self)

    def pdf_clear_annot_ink_list(self):
        """ Class-aware wrapper for `::pdf_clear_annot_ink_list()`."""
        return _mupdf.PdfAnnot_pdf_clear_annot_ink_list(self)

    def pdf_clear_annot_quad_points(self):
        """ Class-aware wrapper for `::pdf_clear_annot_quad_points()`."""
        return _mupdf.PdfAnnot_pdf_clear_annot_quad_points(self)

    def pdf_clear_annot_vertices(self):
        """ Class-aware wrapper for `::pdf_clear_annot_vertices()`."""
        return _mupdf.PdfAnnot_pdf_clear_annot_vertices(self)

    def pdf_clear_signature(self):
        """ Class-aware wrapper for `::pdf_clear_signature()`."""
        return _mupdf.PdfAnnot_pdf_clear_signature(self)

    def pdf_dirty_annot(self):
        """ Class-aware wrapper for `::pdf_dirty_annot()`."""
        return _mupdf.PdfAnnot_pdf_dirty_annot(self)

    def pdf_edit_text_field_value(self, value, change, selStart, selEnd, newvalue):
        """
        Class-aware wrapper for `::pdf_edit_text_field_value()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_edit_text_field_value(const char *value, const char *change)` => `(int, int selStart, int selEnd, char *newvalue)`
        """
        return _mupdf.PdfAnnot_pdf_edit_text_field_value(self, value, change, selStart, selEnd, newvalue)

    def pdf_get_widget_editing_state(self):
        """ Class-aware wrapper for `::pdf_get_widget_editing_state()`."""
        return _mupdf.PdfAnnot_pdf_get_widget_editing_state(self)

    def pdf_new_display_list_from_annot(self):
        """ Class-aware wrapper for `::pdf_new_display_list_from_annot()`."""
        return _mupdf.PdfAnnot_pdf_new_display_list_from_annot(self)

    def pdf_new_pixmap_from_annot(self, ctm, cs, seps, alpha):
        """ Class-aware wrapper for `::pdf_new_pixmap_from_annot()`."""
        return _mupdf.PdfAnnot_pdf_new_pixmap_from_annot(self, ctm, cs, seps, alpha)

    def pdf_next_annot(self):
        """ Class-aware wrapper for `::pdf_next_annot()`."""
        return _mupdf.PdfAnnot_pdf_next_annot(self)

    def pdf_next_widget(self):
        """ Class-aware wrapper for `::pdf_next_widget()`."""
        return _mupdf.PdfAnnot_pdf_next_widget(self)

    def pdf_run_annot(self, dev, ctm, cookie):
        """ Class-aware wrapper for `::pdf_run_annot()`."""
        return _mupdf.PdfAnnot_pdf_run_annot(self, dev, ctm, cookie)

    def pdf_set_annot_active(self, active):
        """ Class-aware wrapper for `::pdf_set_annot_active()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_active(self, active)

    def pdf_set_annot_appearance(self, appearance, state, ctm, bbox, res, contents):
        """ Class-aware wrapper for `::pdf_set_annot_appearance()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_appearance(self, appearance, state, ctm, bbox, res, contents)

    def pdf_set_annot_appearance_from_display_list(self, appearance, state, ctm, list):
        """ Class-aware wrapper for `::pdf_set_annot_appearance_from_display_list()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_appearance_from_display_list(self, appearance, state, ctm, list)

    def pdf_set_annot_author(self, author):
        """ Class-aware wrapper for `::pdf_set_annot_author()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_author(self, author)

    def pdf_set_annot_border(self, width):
        """ Class-aware wrapper for `::pdf_set_annot_border()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_border(self, width)

    def pdf_set_annot_border_effect(self, effect):
        """ Class-aware wrapper for `::pdf_set_annot_border_effect()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_border_effect(self, effect)

    def pdf_set_annot_border_effect_intensity(self, intensity):
        """ Class-aware wrapper for `::pdf_set_annot_border_effect_intensity()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_border_effect_intensity(self, intensity)

    def pdf_set_annot_border_style(self, style):
        """ Class-aware wrapper for `::pdf_set_annot_border_style()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_border_style(self, style)

    def pdf_set_annot_border_width(self, width):
        """ Class-aware wrapper for `::pdf_set_annot_border_width()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_border_width(self, width)

    def pdf_set_annot_color(self, n, color):
        """ Class-aware wrapper for `::pdf_set_annot_color()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_color(self, n, color)

    def pdf_set_annot_contents(self, text):
        """ Class-aware wrapper for `::pdf_set_annot_contents()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_contents(self, text)

    def pdf_set_annot_creation_date(self, time):
        """ Class-aware wrapper for `::pdf_set_annot_creation_date()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_creation_date(self, time)

    def pdf_set_annot_default_appearance(self, font, size, n, color):
        """ Class-aware wrapper for `::pdf_set_annot_default_appearance()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_default_appearance(self, font, size, n, color)

    def pdf_set_annot_filespec(self, obj):
        """ Class-aware wrapper for `::pdf_set_annot_filespec()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_filespec(self, obj)

    def pdf_set_annot_flags(self, flags):
        """ Class-aware wrapper for `::pdf_set_annot_flags()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_flags(self, flags)

    def pdf_set_annot_hidden_for_editing(self, hidden):
        """ Class-aware wrapper for `::pdf_set_annot_hidden_for_editing()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_hidden_for_editing(self, hidden)

    def pdf_set_annot_hot(self, hot):
        """ Class-aware wrapper for `::pdf_set_annot_hot()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_hot(self, hot)

    def pdf_set_annot_icon_name(self, name):
        """ Class-aware wrapper for `::pdf_set_annot_icon_name()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_icon_name(self, name)

    def pdf_set_annot_ink_list(self, n, count, v):
        """ Class-aware wrapper for `::pdf_set_annot_ink_list()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_ink_list(self, n, count, v)

    def pdf_set_annot_interior_color(self, n, color):
        """ Class-aware wrapper for `::pdf_set_annot_interior_color()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_interior_color(self, n, color)

    def pdf_set_annot_is_open(self, is_open):
        """ Class-aware wrapper for `::pdf_set_annot_is_open()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_is_open(self, is_open)

    def pdf_set_annot_language(self, lang):
        """ Class-aware wrapper for `::pdf_set_annot_language()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_language(self, lang)

    def pdf_set_annot_line(self, a, b):
        """ Class-aware wrapper for `::pdf_set_annot_line()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_line(self, a, b)

    def pdf_set_annot_line_end_style(self, e):
        """ Class-aware wrapper for `::pdf_set_annot_line_end_style()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_line_end_style(self, e)

    def pdf_set_annot_line_ending_styles(self, start_style, end_style):
        """ Class-aware wrapper for `::pdf_set_annot_line_ending_styles()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_line_ending_styles(self, start_style, end_style)

    def pdf_set_annot_line_start_style(self, s):
        """ Class-aware wrapper for `::pdf_set_annot_line_start_style()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_line_start_style(self, s)

    def pdf_set_annot_modification_date(self, time):
        """ Class-aware wrapper for `::pdf_set_annot_modification_date()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_modification_date(self, time)

    def pdf_set_annot_opacity(self, opacity):
        """ Class-aware wrapper for `::pdf_set_annot_opacity()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_opacity(self, opacity)

    def pdf_set_annot_popup(self, rect):
        """ Class-aware wrapper for `::pdf_set_annot_popup()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_popup(self, rect)

    def pdf_set_annot_quad_points(self, n, qv):
        """ Class-aware wrapper for `::pdf_set_annot_quad_points()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_quad_points(self, n, qv)

    def pdf_set_annot_quadding(self, q):
        """ Class-aware wrapper for `::pdf_set_annot_quadding()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_quadding(self, q)

    def pdf_set_annot_rect(self, rect):
        """ Class-aware wrapper for `::pdf_set_annot_rect()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_rect(self, rect)

    def pdf_set_annot_resynthesised(self):
        """ Class-aware wrapper for `::pdf_set_annot_resynthesised()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_resynthesised(self)

    def pdf_set_annot_stamp_image(self, image):
        """ Class-aware wrapper for `::pdf_set_annot_stamp_image()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_stamp_image(self, image)

    def pdf_set_annot_vertex(self, i, p):
        """ Class-aware wrapper for `::pdf_set_annot_vertex()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_vertex(self, i, p)

    def pdf_set_annot_vertices(self, n, v):
        """ Class-aware wrapper for `::pdf_set_annot_vertices()`."""
        return _mupdf.PdfAnnot_pdf_set_annot_vertices(self, n, v)

    def pdf_set_choice_field_value(self, value):
        """ Class-aware wrapper for `::pdf_set_choice_field_value()`."""
        return _mupdf.PdfAnnot_pdf_set_choice_field_value(self, value)

    def pdf_set_text_field_value(self, value):
        """ Class-aware wrapper for `::pdf_set_text_field_value()`."""
        return _mupdf.PdfAnnot_pdf_set_text_field_value(self, value)

    def pdf_set_widget_editing_state(self, editing):
        """ Class-aware wrapper for `::pdf_set_widget_editing_state()`."""
        return _mupdf.PdfAnnot_pdf_set_widget_editing_state(self, editing)

    def pdf_sign_signature(self, signer, appearance_flags, graphic, reason, location):
        """ Class-aware wrapper for `::pdf_sign_signature()`."""
        return _mupdf.PdfAnnot_pdf_sign_signature(self, signer, appearance_flags, graphic, reason, location)

    def pdf_sign_signature_with_appearance(self, signer, date, disp_list):
        """ Class-aware wrapper for `::pdf_sign_signature_with_appearance()`."""
        return _mupdf.PdfAnnot_pdf_sign_signature_with_appearance(self, signer, date, disp_list)

    def pdf_text_widget_format(self):
        """ Class-aware wrapper for `::pdf_text_widget_format()`."""
        return _mupdf.PdfAnnot_pdf_text_widget_format(self)

    def pdf_text_widget_max_len(self):
        """ Class-aware wrapper for `::pdf_text_widget_max_len()`."""
        return _mupdf.PdfAnnot_pdf_text_widget_max_len(self)

    def pdf_toggle_widget(self):
        """ Class-aware wrapper for `::pdf_toggle_widget()`."""
        return _mupdf.PdfAnnot_pdf_toggle_widget(self)

    def pdf_update_annot(self):
        """ Class-aware wrapper for `::pdf_update_annot()`."""
        return _mupdf.PdfAnnot_pdf_update_annot(self)

    def pdf_update_widget(self):
        """ Class-aware wrapper for `::pdf_update_widget()`."""
        return _mupdf.PdfAnnot_pdf_update_widget(self)

    def pdf_validate_signature(self):
        """ Class-aware wrapper for `::pdf_validate_signature()`."""
        return _mupdf.PdfAnnot_pdf_validate_signature(self)

    def pdf_widget_is_readonly(self):
        """ Class-aware wrapper for `::pdf_widget_is_readonly()`."""
        return _mupdf.PdfAnnot_pdf_widget_is_readonly(self)

    def pdf_widget_is_signed(self):
        """ Class-aware wrapper for `::pdf_widget_is_signed()`."""
        return _mupdf.PdfAnnot_pdf_widget_is_signed(self)

    def pdf_widget_type(self):
        """ Class-aware wrapper for `::pdf_widget_type()`."""
        return _mupdf.PdfAnnot_pdf_widget_type(self)

    def __init__(self, *args):
        """
        *Overload 1:*
        Copy constructor using `pdf_keep_annot()`.

        |

        *Overload 2:*
        Constructor using raw copy of pre-existing `::pdf_annot`.

        |

        *Overload 3:*
        Constructor using raw copy of pre-existing `::pdf_annot`.
        """
        _mupdf.PdfAnnot_swiginit(self, _mupdf.new_PdfAnnot(*args))
    __swig_destroy__ = _mupdf.delete_PdfAnnot

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.PdfAnnot_m_internal_value(self)
    m_internal = property(_mupdf.PdfAnnot_m_internal_get, _mupdf.PdfAnnot_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.PdfAnnot_s_num_instances_get, _mupdf.PdfAnnot_s_num_instances_set)