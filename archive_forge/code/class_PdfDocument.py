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
class PdfDocument(object):
    """ Wrapper class for struct `pdf_document`."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def pdf_abandon_operation(self):
        """ Class-aware wrapper for `::pdf_abandon_operation()`."""
        return _mupdf.PdfDocument_pdf_abandon_operation(self)

    def pdf_add_cid_font(self, font):
        """ Class-aware wrapper for `::pdf_add_cid_font()`."""
        return _mupdf.PdfDocument_pdf_add_cid_font(self, font)

    def pdf_add_cjk_font(self, font, script, wmode, serif):
        """ Class-aware wrapper for `::pdf_add_cjk_font()`."""
        return _mupdf.PdfDocument_pdf_add_cjk_font(self, font, script, wmode, serif)

    def pdf_add_embedded_file(self, filename, mimetype, contents, created, modifed, add_checksum):
        """ Class-aware wrapper for `::pdf_add_embedded_file()`."""
        return _mupdf.PdfDocument_pdf_add_embedded_file(self, filename, mimetype, contents, created, modifed, add_checksum)

    def pdf_add_image(self, image):
        """ Class-aware wrapper for `::pdf_add_image()`."""
        return _mupdf.PdfDocument_pdf_add_image(self, image)

    def pdf_add_journal_fragment(self, parent, copy, copy_stream, newobj):
        """ Class-aware wrapper for `::pdf_add_journal_fragment()`."""
        return _mupdf.PdfDocument_pdf_add_journal_fragment(self, parent, copy, copy_stream, newobj)

    def pdf_add_new_array(self, initial):
        """ Class-aware wrapper for `::pdf_add_new_array()`."""
        return _mupdf.PdfDocument_pdf_add_new_array(self, initial)

    def pdf_add_new_dict(self, initial):
        """ Class-aware wrapper for `::pdf_add_new_dict()`."""
        return _mupdf.PdfDocument_pdf_add_new_dict(self, initial)

    def pdf_add_object(self, obj):
        """ Class-aware wrapper for `::pdf_add_object()`."""
        return _mupdf.PdfDocument_pdf_add_object(self, obj)

    def pdf_add_page(self, mediabox, rotate, resources, contents):
        """ Class-aware wrapper for `::pdf_add_page()`."""
        return _mupdf.PdfDocument_pdf_add_page(self, mediabox, rotate, resources, contents)

    def pdf_add_simple_font(self, font, encoding):
        """ Class-aware wrapper for `::pdf_add_simple_font()`."""
        return _mupdf.PdfDocument_pdf_add_simple_font(self, font, encoding)

    def pdf_add_stream(self, buf, obj, compressed):
        """ Class-aware wrapper for `::pdf_add_stream()`."""
        return _mupdf.PdfDocument_pdf_add_stream(self, buf, obj, compressed)

    def pdf_add_substitute_font(self, font):
        """ Class-aware wrapper for `::pdf_add_substitute_font()`."""
        return _mupdf.PdfDocument_pdf_add_substitute_font(self, font)

    def pdf_annot_field_event_keystroke(self, annot, evt):
        """ Class-aware wrapper for `::pdf_annot_field_event_keystroke()`."""
        return _mupdf.PdfDocument_pdf_annot_field_event_keystroke(self, annot, evt)

    def pdf_authenticate_password(self, pw):
        """ Class-aware wrapper for `::pdf_authenticate_password()`."""
        return _mupdf.PdfDocument_pdf_authenticate_password(self, pw)

    def pdf_bake_document(self, bake_annots, bake_widgets):
        """ Class-aware wrapper for `::pdf_bake_document()`."""
        return _mupdf.PdfDocument_pdf_bake_document(self, bake_annots, bake_widgets)

    def pdf_begin_implicit_operation(self):
        """ Class-aware wrapper for `::pdf_begin_implicit_operation()`."""
        return _mupdf.PdfDocument_pdf_begin_implicit_operation(self)

    def pdf_begin_operation(self, operation):
        """ Class-aware wrapper for `::pdf_begin_operation()`."""
        return _mupdf.PdfDocument_pdf_begin_operation(self, operation)

    def pdf_calculate_form(self):
        """ Class-aware wrapper for `::pdf_calculate_form()`."""
        return _mupdf.PdfDocument_pdf_calculate_form(self)

    def pdf_can_be_saved_incrementally(self):
        """ Class-aware wrapper for `::pdf_can_be_saved_incrementally()`."""
        return _mupdf.PdfDocument_pdf_can_be_saved_incrementally(self)

    def pdf_can_redo(self):
        """ Class-aware wrapper for `::pdf_can_redo()`."""
        return _mupdf.PdfDocument_pdf_can_redo(self)

    def pdf_can_undo(self):
        """ Class-aware wrapper for `::pdf_can_undo()`."""
        return _mupdf.PdfDocument_pdf_can_undo(self)

    def pdf_clear_xref(self):
        """ Class-aware wrapper for `::pdf_clear_xref()`."""
        return _mupdf.PdfDocument_pdf_clear_xref(self)

    def pdf_clear_xref_to_mark(self):
        """ Class-aware wrapper for `::pdf_clear_xref_to_mark()`."""
        return _mupdf.PdfDocument_pdf_clear_xref_to_mark(self)

    def pdf_count_layer_config_ui(self):
        """ Class-aware wrapper for `::pdf_count_layer_config_ui()`."""
        return _mupdf.PdfDocument_pdf_count_layer_config_ui(self)

    def pdf_count_layer_configs(self):
        """ Class-aware wrapper for `::pdf_count_layer_configs()`."""
        return _mupdf.PdfDocument_pdf_count_layer_configs(self)

    def pdf_count_layers(self):
        """ Class-aware wrapper for `::pdf_count_layers()`."""
        return _mupdf.PdfDocument_pdf_count_layers(self)

    def pdf_count_objects(self):
        """ Class-aware wrapper for `::pdf_count_objects()`."""
        return _mupdf.PdfDocument_pdf_count_objects(self)

    def pdf_count_pages(self):
        """ Class-aware wrapper for `::pdf_count_pages()`."""
        return _mupdf.PdfDocument_pdf_count_pages(self)

    def pdf_count_q_balance(self, res, stm, underflow, overflow):
        """
        Class-aware wrapper for `::pdf_count_q_balance()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_count_q_balance(::pdf_obj *res, ::pdf_obj *stm)` => `(int underflow, int overflow)`
        """
        return _mupdf.PdfDocument_pdf_count_q_balance(self, res, stm, underflow, overflow)

    def pdf_count_signatures(self):
        """ Class-aware wrapper for `::pdf_count_signatures()`."""
        return _mupdf.PdfDocument_pdf_count_signatures(self)

    def pdf_count_unsaved_versions(self):
        """ Class-aware wrapper for `::pdf_count_unsaved_versions()`."""
        return _mupdf.PdfDocument_pdf_count_unsaved_versions(self)

    def pdf_count_versions(self):
        """ Class-aware wrapper for `::pdf_count_versions()`."""
        return _mupdf.PdfDocument_pdf_count_versions(self)

    def pdf_create_field_name(self, prefix, buf, len):
        """ Class-aware wrapper for `::pdf_create_field_name()`."""
        return _mupdf.PdfDocument_pdf_create_field_name(self, prefix, buf, len)

    def pdf_create_object(self):
        """ Class-aware wrapper for `::pdf_create_object()`."""
        return _mupdf.PdfDocument_pdf_create_object(self)

    def pdf_debug_doc_changes(self):
        """ Class-aware wrapper for `::pdf_debug_doc_changes()`."""
        return _mupdf.PdfDocument_pdf_debug_doc_changes(self)

    def pdf_delete_object(self, num):
        """ Class-aware wrapper for `::pdf_delete_object()`."""
        return _mupdf.PdfDocument_pdf_delete_object(self, num)

    def pdf_delete_page(self, number):
        """ Class-aware wrapper for `::pdf_delete_page()`."""
        return _mupdf.PdfDocument_pdf_delete_page(self, number)

    def pdf_delete_page_labels(self, index):
        """ Class-aware wrapper for `::pdf_delete_page_labels()`."""
        return _mupdf.PdfDocument_pdf_delete_page_labels(self, index)

    def pdf_delete_page_range(self, start, end):
        """ Class-aware wrapper for `::pdf_delete_page_range()`."""
        return _mupdf.PdfDocument_pdf_delete_page_range(self, start, end)

    def pdf_deselect_layer_config_ui(self, ui):
        """ Class-aware wrapper for `::pdf_deselect_layer_config_ui()`."""
        return _mupdf.PdfDocument_pdf_deselect_layer_config_ui(self, ui)

    def pdf_deserialise_journal(self, stm):
        """ Class-aware wrapper for `::pdf_deserialise_journal()`."""
        return _mupdf.PdfDocument_pdf_deserialise_journal(self, stm)

    def pdf_disable_js(self):
        """ Class-aware wrapper for `::pdf_disable_js()`."""
        return _mupdf.PdfDocument_pdf_disable_js(self)

    def pdf_doc_was_linearized(self):
        """ Class-aware wrapper for `::pdf_doc_was_linearized()`."""
        return _mupdf.PdfDocument_pdf_doc_was_linearized(self)

    def pdf_document_event_did_print(self):
        """ Class-aware wrapper for `::pdf_document_event_did_print()`."""
        return _mupdf.PdfDocument_pdf_document_event_did_print(self)

    def pdf_document_event_did_save(self):
        """ Class-aware wrapper for `::pdf_document_event_did_save()`."""
        return _mupdf.PdfDocument_pdf_document_event_did_save(self)

    def pdf_document_event_will_close(self):
        """ Class-aware wrapper for `::pdf_document_event_will_close()`."""
        return _mupdf.PdfDocument_pdf_document_event_will_close(self)

    def pdf_document_event_will_print(self):
        """ Class-aware wrapper for `::pdf_document_event_will_print()`."""
        return _mupdf.PdfDocument_pdf_document_event_will_print(self)

    def pdf_document_event_will_save(self):
        """ Class-aware wrapper for `::pdf_document_event_will_save()`."""
        return _mupdf.PdfDocument_pdf_document_event_will_save(self)

    def pdf_document_output_intent(self):
        """ Class-aware wrapper for `::pdf_document_output_intent()`."""
        return _mupdf.PdfDocument_pdf_document_output_intent(self)

    def pdf_document_permissions(self):
        """ Class-aware wrapper for `::pdf_document_permissions()`."""
        return _mupdf.PdfDocument_pdf_document_permissions(self)

    def pdf_empty_store(self):
        """ Class-aware wrapper for `::pdf_empty_store()`."""
        return _mupdf.PdfDocument_pdf_empty_store(self)

    def pdf_enable_journal(self):
        """ Class-aware wrapper for `::pdf_enable_journal()`."""
        return _mupdf.PdfDocument_pdf_enable_journal(self)

    def pdf_enable_js(self):
        """ Class-aware wrapper for `::pdf_enable_js()`."""
        return _mupdf.PdfDocument_pdf_enable_js(self)

    def pdf_enable_layer(self, layer, enabled):
        """ Class-aware wrapper for `::pdf_enable_layer()`."""
        return _mupdf.PdfDocument_pdf_enable_layer(self, layer, enabled)

    def pdf_end_operation(self):
        """ Class-aware wrapper for `::pdf_end_operation()`."""
        return _mupdf.PdfDocument_pdf_end_operation(self)

    def pdf_ensure_solid_xref(self, num):
        """ Class-aware wrapper for `::pdf_ensure_solid_xref()`."""
        return _mupdf.PdfDocument_pdf_ensure_solid_xref(self, num)

    def pdf_event_issue_alert(self, evt):
        """ Class-aware wrapper for `::pdf_event_issue_alert()`."""
        return _mupdf.PdfDocument_pdf_event_issue_alert(self, evt)

    def pdf_event_issue_exec_menu_item(self, item):
        """ Class-aware wrapper for `::pdf_event_issue_exec_menu_item()`."""
        return _mupdf.PdfDocument_pdf_event_issue_exec_menu_item(self, item)

    def pdf_event_issue_launch_url(self, url, new_frame):
        """ Class-aware wrapper for `::pdf_event_issue_launch_url()`."""
        return _mupdf.PdfDocument_pdf_event_issue_launch_url(self, url, new_frame)

    def pdf_event_issue_mail_doc(self, evt):
        """ Class-aware wrapper for `::pdf_event_issue_mail_doc()`."""
        return _mupdf.PdfDocument_pdf_event_issue_mail_doc(self, evt)

    def pdf_event_issue_print(self):
        """ Class-aware wrapper for `::pdf_event_issue_print()`."""
        return _mupdf.PdfDocument_pdf_event_issue_print(self)

    def pdf_field_event_calculate(self, field):
        """ Class-aware wrapper for `::pdf_field_event_calculate()`."""
        return _mupdf.PdfDocument_pdf_field_event_calculate(self, field)

    def pdf_field_event_format(self, field):
        """ Class-aware wrapper for `::pdf_field_event_format()`."""
        return _mupdf.PdfDocument_pdf_field_event_format(self, field)

    def pdf_field_event_keystroke(self, field, evt):
        """ Class-aware wrapper for `::pdf_field_event_keystroke()`."""
        return _mupdf.PdfDocument_pdf_field_event_keystroke(self, field, evt)

    def pdf_field_event_validate(self, field, value, newvalue):
        """
        Class-aware wrapper for `::pdf_field_event_validate()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_field_event_validate(::pdf_obj *field, const char *value)` => `(int, char *newvalue)`
        """
        return _mupdf.PdfDocument_pdf_field_event_validate(self, field, value, newvalue)

    def pdf_field_reset(self, field):
        """ Class-aware wrapper for `::pdf_field_reset()`."""
        return _mupdf.PdfDocument_pdf_field_reset(self, field)

    def pdf_filter_annot_contents(self, annot, options):
        """ Class-aware wrapper for `::pdf_filter_annot_contents()`."""
        return _mupdf.PdfDocument_pdf_filter_annot_contents(self, annot, options)

    def pdf_filter_page_contents(self, page, options):
        """ Class-aware wrapper for `::pdf_filter_page_contents()`."""
        return _mupdf.PdfDocument_pdf_filter_page_contents(self, page, options)

    def pdf_find_font_resource(self, type, encoding, item, key):
        """ Class-aware wrapper for `::pdf_find_font_resource()`."""
        return _mupdf.PdfDocument_pdf_find_font_resource(self, type, encoding, item, key)

    def pdf_find_version_for_obj(self, obj):
        """ Class-aware wrapper for `::pdf_find_version_for_obj()`."""
        return _mupdf.PdfDocument_pdf_find_version_for_obj(self, obj)

    def pdf_forget_xref(self):
        """ Class-aware wrapper for `::pdf_forget_xref()`."""
        return _mupdf.PdfDocument_pdf_forget_xref(self)

    def pdf_get_doc_event_callback_data(self):
        """ Class-aware wrapper for `::pdf_get_doc_event_callback_data()`."""
        return _mupdf.PdfDocument_pdf_get_doc_event_callback_data(self)

    def pdf_graft_object(self, obj):
        """ Class-aware wrapper for `::pdf_graft_object()`."""
        return _mupdf.PdfDocument_pdf_graft_object(self, obj)

    def pdf_graft_page(self, page_to, src, page_from):
        """ Class-aware wrapper for `::pdf_graft_page()`."""
        return _mupdf.PdfDocument_pdf_graft_page(self, page_to, src, page_from)

    def pdf_has_permission(self, p):
        """ Class-aware wrapper for `::pdf_has_permission()`."""
        return _mupdf.PdfDocument_pdf_has_permission(self, p)

    def pdf_has_unsaved_changes(self):
        """ Class-aware wrapper for `::pdf_has_unsaved_changes()`."""
        return _mupdf.PdfDocument_pdf_has_unsaved_changes(self)

    def pdf_has_unsaved_sigs(self):
        """ Class-aware wrapper for `::pdf_has_unsaved_sigs()`."""
        return _mupdf.PdfDocument_pdf_has_unsaved_sigs(self)

    def pdf_insert_font_resource(self, key, obj):
        """ Class-aware wrapper for `::pdf_insert_font_resource()`."""
        return _mupdf.PdfDocument_pdf_insert_font_resource(self, key, obj)

    def pdf_insert_page(self, at, page):
        """ Class-aware wrapper for `::pdf_insert_page()`."""
        return _mupdf.PdfDocument_pdf_insert_page(self, at, page)

    def pdf_invalidate_xfa(self):
        """ Class-aware wrapper for `::pdf_invalidate_xfa()`."""
        return _mupdf.PdfDocument_pdf_invalidate_xfa(self)

    def pdf_is_local_object(self, obj):
        """ Class-aware wrapper for `::pdf_is_local_object()`."""
        return _mupdf.PdfDocument_pdf_is_local_object(self, obj)

    def pdf_is_ocg_hidden(self, rdb, usage, ocg):
        """ Class-aware wrapper for `::pdf_is_ocg_hidden()`."""
        return _mupdf.PdfDocument_pdf_is_ocg_hidden(self, rdb, usage, ocg)

    def pdf_js_set_console(self, console, user):
        """ Class-aware wrapper for `::pdf_js_set_console()`."""
        return _mupdf.PdfDocument_pdf_js_set_console(self, console, user)

    def pdf_js_supported(self):
        """ Class-aware wrapper for `::pdf_js_supported()`."""
        return _mupdf.PdfDocument_pdf_js_supported(self)

    def pdf_layer_config_info(self, config_num, info):
        """ Class-aware wrapper for `::pdf_layer_config_info()`."""
        return _mupdf.PdfDocument_pdf_layer_config_info(self, config_num, info)

    def pdf_layer_config_ui_info(self, ui, info):
        """ Class-aware wrapper for `::pdf_layer_config_ui_info()`."""
        return _mupdf.PdfDocument_pdf_layer_config_ui_info(self, ui, info)

    def pdf_layer_is_enabled(self, layer):
        """ Class-aware wrapper for `::pdf_layer_is_enabled()`."""
        return _mupdf.PdfDocument_pdf_layer_is_enabled(self, layer)

    def pdf_layer_name(self, layer):
        """ Class-aware wrapper for `::pdf_layer_name()`."""
        return _mupdf.PdfDocument_pdf_layer_name(self, layer)

    def pdf_load_compressed_inline_image(self, dict, length, cstm, indexed, image):
        """ Class-aware wrapper for `::pdf_load_compressed_inline_image()`."""
        return _mupdf.PdfDocument_pdf_load_compressed_inline_image(self, dict, length, cstm, indexed, image)

    def pdf_load_compressed_stream(self, num, worst_case):
        """ Class-aware wrapper for `::pdf_load_compressed_stream()`."""
        return _mupdf.PdfDocument_pdf_load_compressed_stream(self, num, worst_case)

    def pdf_load_default_colorspaces(self, page):
        """ Class-aware wrapper for `::pdf_load_default_colorspaces()`."""
        return _mupdf.PdfDocument_pdf_load_default_colorspaces(self, page)

    def pdf_load_embedded_cmap(self, ref):
        """ Class-aware wrapper for `::pdf_load_embedded_cmap()`."""
        return _mupdf.PdfDocument_pdf_load_embedded_cmap(self, ref)

    def pdf_load_image(self, obj):
        """ Class-aware wrapper for `::pdf_load_image()`."""
        return _mupdf.PdfDocument_pdf_load_image(self, obj)

    def pdf_load_inline_image(self, rdb, dict, file):
        """ Class-aware wrapper for `::pdf_load_inline_image()`."""
        return _mupdf.PdfDocument_pdf_load_inline_image(self, rdb, dict, file)

    def pdf_load_journal(self, filename):
        """ Class-aware wrapper for `::pdf_load_journal()`."""
        return _mupdf.PdfDocument_pdf_load_journal(self, filename)

    def pdf_load_link_annots(self, arg_1, annots, pagenum, page_ctm):
        """ Class-aware wrapper for `::pdf_load_link_annots()`."""
        return _mupdf.PdfDocument_pdf_load_link_annots(self, arg_1, annots, pagenum, page_ctm)

    def pdf_load_name_tree(self, which):
        """ Class-aware wrapper for `::pdf_load_name_tree()`."""
        return _mupdf.PdfDocument_pdf_load_name_tree(self, which)

    def pdf_load_object(self, num):
        """
        Class-aware wrapper for `::pdf_load_object()`.
        	Load a given object.

        	This can cause xref reorganisations (solidifications etc) due to
        	repairs, so all held pdf_xref_entries should be considered
        	invalid after this call (other than the returned one).
        """
        return _mupdf.PdfDocument_pdf_load_object(self, num)

    def pdf_load_outline(self):
        """ Class-aware wrapper for `::pdf_load_outline()`."""
        return _mupdf.PdfDocument_pdf_load_outline(self)

    def pdf_load_page(self, number):
        """ Class-aware wrapper for `::pdf_load_page()`."""
        return _mupdf.PdfDocument_pdf_load_page(self, number)

    def pdf_load_page_tree(self):
        """ Class-aware wrapper for `::pdf_load_page_tree()`."""
        return _mupdf.PdfDocument_pdf_load_page_tree(self)

    def pdf_load_pattern(self, obj):
        """ Class-aware wrapper for `::pdf_load_pattern()`."""
        return _mupdf.PdfDocument_pdf_load_pattern(self, obj)

    def pdf_load_raw_stream_number(self, num):
        """ Class-aware wrapper for `::pdf_load_raw_stream_number()`."""
        return _mupdf.PdfDocument_pdf_load_raw_stream_number(self, num)

    def pdf_load_shading(self, obj):
        """ Class-aware wrapper for `::pdf_load_shading()`."""
        return _mupdf.PdfDocument_pdf_load_shading(self, obj)

    def pdf_load_stream_number(self, num):
        """ Class-aware wrapper for `::pdf_load_stream_number()`."""
        return _mupdf.PdfDocument_pdf_load_stream_number(self, num)

    def pdf_load_to_unicode(self, font, strings, collection, cmapstm):
        """
        Class-aware wrapper for `::pdf_load_to_unicode()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_load_to_unicode(::pdf_font_desc *font, char *collection, ::pdf_obj *cmapstm)` => const char *strings
        """
        return _mupdf.PdfDocument_pdf_load_to_unicode(self, font, strings, collection, cmapstm)

    def pdf_load_type3_glyphs(self, fontdesc):
        """ Class-aware wrapper for `::pdf_load_type3_glyphs()`."""
        return _mupdf.PdfDocument_pdf_load_type3_glyphs(self, fontdesc)

    def pdf_load_unencrypted_object(self, num):
        """ Class-aware wrapper for `::pdf_load_unencrypted_object()`."""
        return _mupdf.PdfDocument_pdf_load_unencrypted_object(self, num)

    def pdf_lookup_dest(self, needle):
        """ Class-aware wrapper for `::pdf_lookup_dest()`."""
        return _mupdf.PdfDocument_pdf_lookup_dest(self, needle)

    def pdf_lookup_metadata(self, key, ptr, size):
        """ Class-aware wrapper for `::pdf_lookup_metadata()`."""
        return _mupdf.PdfDocument_pdf_lookup_metadata(self, key, ptr, size)

    def pdf_lookup_metadata2(self, key):
        """
        Class-aware wrapper for `::pdf_lookup_metadata2()`.
        C++ alternative to `pdf_lookup_metadata()` that returns a `std::string`
        or calls `fz_throw()` if not found.
        """
        return _mupdf.PdfDocument_pdf_lookup_metadata2(self, key)

    def pdf_lookup_name(self, which, needle):
        """ Class-aware wrapper for `::pdf_lookup_name()`."""
        return _mupdf.PdfDocument_pdf_lookup_name(self, which, needle)

    def pdf_lookup_page_loc(self, needle, parentp, indexp):
        """
        Class-aware wrapper for `::pdf_lookup_page_loc()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_lookup_page_loc(int needle, ::pdf_obj **parentp)` => `(pdf_obj *, int indexp)`
        """
        return _mupdf.PdfDocument_pdf_lookup_page_loc(self, needle, parentp, indexp)

    def pdf_lookup_page_number(self, pageobj):
        """ Class-aware wrapper for `::pdf_lookup_page_number()`."""
        return _mupdf.PdfDocument_pdf_lookup_page_number(self, pageobj)

    def pdf_lookup_page_obj(self, needle):
        """ Class-aware wrapper for `::pdf_lookup_page_obj()`."""
        return _mupdf.PdfDocument_pdf_lookup_page_obj(self, needle)

    def pdf_mark_xref(self):
        """ Class-aware wrapper for `::pdf_mark_xref()`."""
        return _mupdf.PdfDocument_pdf_mark_xref(self)

    def pdf_metadata(self):
        """ Class-aware wrapper for `::pdf_metadata()`."""
        return _mupdf.PdfDocument_pdf_metadata(self)

    def pdf_minimize_document(self):
        """ Class-aware wrapper for `::pdf_minimize_document()`."""
        return _mupdf.PdfDocument_pdf_minimize_document(self)

    def pdf_needs_password(self):
        """ Class-aware wrapper for `::pdf_needs_password()`."""
        return _mupdf.PdfDocument_pdf_needs_password(self)

    def pdf_new_action_from_link(self, uri):
        """ Class-aware wrapper for `::pdf_new_action_from_link()`."""
        return _mupdf.PdfDocument_pdf_new_action_from_link(self, uri)

    def pdf_new_array(self, initialcap):
        """ Class-aware wrapper for `::pdf_new_array()`."""
        return _mupdf.PdfDocument_pdf_new_array(self, initialcap)

    def pdf_new_color_filter(self, chain, struct_parents, transform, options, copts):
        """ Class-aware wrapper for `::pdf_new_color_filter()`."""
        return _mupdf.PdfDocument_pdf_new_color_filter(self, chain, struct_parents, transform, options, copts)

    def pdf_new_date(self, time):
        """ Class-aware wrapper for `::pdf_new_date()`."""
        return _mupdf.PdfDocument_pdf_new_date(self, time)

    def pdf_new_dest_from_link(self, uri, is_remote):
        """ Class-aware wrapper for `::pdf_new_dest_from_link()`."""
        return _mupdf.PdfDocument_pdf_new_dest_from_link(self, uri, is_remote)

    def pdf_new_dict(self, initialcap):
        """ Class-aware wrapper for `::pdf_new_dict()`."""
        return _mupdf.PdfDocument_pdf_new_dict(self, initialcap)

    def pdf_new_graft_map(self):
        """ Class-aware wrapper for `::pdf_new_graft_map()`."""
        return _mupdf.PdfDocument_pdf_new_graft_map(self)

    def pdf_new_indirect(self, num, gen):
        """ Class-aware wrapper for `::pdf_new_indirect()`."""
        return _mupdf.PdfDocument_pdf_new_indirect(self, num, gen)

    def pdf_new_matrix(self, mtx):
        """ Class-aware wrapper for `::pdf_new_matrix()`."""
        return _mupdf.PdfDocument_pdf_new_matrix(self, mtx)

    def pdf_new_pdf_device(self, topctm, resources, contents):
        """ Class-aware wrapper for `::pdf_new_pdf_device()`."""
        return _mupdf.PdfDocument_pdf_new_pdf_device(self, topctm, resources, contents)

    def pdf_new_rect(self, rect):
        """ Class-aware wrapper for `::pdf_new_rect()`."""
        return _mupdf.PdfDocument_pdf_new_rect(self, rect)

    def pdf_new_run_processor(self, dev, ctm, struct_parent, usage, gstate, default_cs, cookie):
        """ Class-aware wrapper for `::pdf_new_run_processor()`."""
        return _mupdf.PdfDocument_pdf_new_run_processor(self, dev, ctm, struct_parent, usage, gstate, default_cs, cookie)

    def pdf_new_sanitize_filter(self, chain, struct_parents, transform, options, sopts):
        """ Class-aware wrapper for `::pdf_new_sanitize_filter()`."""
        return _mupdf.PdfDocument_pdf_new_sanitize_filter(self, chain, struct_parents, transform, options, sopts)

    def pdf_new_xobject(self, bbox, matrix, res, buffer):
        """ Class-aware wrapper for `::pdf_new_xobject()`."""
        return _mupdf.PdfDocument_pdf_new_xobject(self, bbox, matrix, res, buffer)

    def pdf_obj_num_is_stream(self, num):
        """ Class-aware wrapper for `::pdf_obj_num_is_stream()`."""
        return _mupdf.PdfDocument_pdf_obj_num_is_stream(self, num)

    def pdf_open_contents_stream(self, obj):
        """ Class-aware wrapper for `::pdf_open_contents_stream()`."""
        return _mupdf.PdfDocument_pdf_open_contents_stream(self, obj)

    def pdf_open_inline_stream(self, stmobj, length, chain, params):
        """ Class-aware wrapper for `::pdf_open_inline_stream()`."""
        return _mupdf.PdfDocument_pdf_open_inline_stream(self, stmobj, length, chain, params)

    def pdf_open_raw_stream_number(self, num):
        """ Class-aware wrapper for `::pdf_open_raw_stream_number()`."""
        return _mupdf.PdfDocument_pdf_open_raw_stream_number(self, num)

    def pdf_open_stream_number(self, num):
        """ Class-aware wrapper for `::pdf_open_stream_number()`."""
        return _mupdf.PdfDocument_pdf_open_stream_number(self, num)

    def pdf_open_stream_with_offset(self, num, dict, stm_ofs):
        """ Class-aware wrapper for `::pdf_open_stream_with_offset()`."""
        return _mupdf.PdfDocument_pdf_open_stream_with_offset(self, num, dict, stm_ofs)

    def pdf_page_label(self, page, buf, size):
        """ Class-aware wrapper for `::pdf_page_label()`."""
        return _mupdf.PdfDocument_pdf_page_label(self, page, buf, size)

    def pdf_page_write(self, mediabox, presources, pcontents):
        """
        Class-aware wrapper for `::pdf_page_write()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_page_write(::fz_rect mediabox, ::pdf_obj **presources, ::fz_buffer **pcontents)` => `(fz_device *)`
        """
        return _mupdf.PdfDocument_pdf_page_write(self, mediabox, presources, pcontents)

    def pdf_parse_array(self, f, buf):
        """ Class-aware wrapper for `::pdf_parse_array()`."""
        return _mupdf.PdfDocument_pdf_parse_array(self, f, buf)

    def pdf_parse_dict(self, f, buf):
        """ Class-aware wrapper for `::pdf_parse_dict()`."""
        return _mupdf.PdfDocument_pdf_parse_dict(self, f, buf)

    def pdf_parse_ind_obj(self, f, num, gen, stm_ofs, try_repair):
        """
        Class-aware wrapper for `::pdf_parse_ind_obj()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_parse_ind_obj(::fz_stream *f)` => `(pdf_obj *, int num, int gen, int64_t stm_ofs, int try_repair)`
        """
        return _mupdf.PdfDocument_pdf_parse_ind_obj(self, f, num, gen, stm_ofs, try_repair)

    def pdf_parse_journal_obj(self, stm, onum, ostm, newobj):
        """
        Class-aware wrapper for `::pdf_parse_journal_obj()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_parse_journal_obj(::fz_stream *stm, ::fz_buffer **ostm)` => `(pdf_obj *, int onum, int newobj)`
        """
        return _mupdf.PdfDocument_pdf_parse_journal_obj(self, stm, onum, ostm, newobj)

    def pdf_parse_stm_obj(self, f, buf):
        """ Class-aware wrapper for `::pdf_parse_stm_obj()`."""
        return _mupdf.PdfDocument_pdf_parse_stm_obj(self, f, buf)

    def pdf_progressive_advance(self, pagenum):
        """ Class-aware wrapper for `::pdf_progressive_advance()`."""
        return _mupdf.PdfDocument_pdf_progressive_advance(self, pagenum)

    def pdf_purge_local_font_resources(self):
        """ Class-aware wrapper for `::pdf_purge_local_font_resources()`."""
        return _mupdf.PdfDocument_pdf_purge_local_font_resources(self)

    def pdf_purge_locals_from_store(self):
        """ Class-aware wrapper for `::pdf_purge_locals_from_store()`."""
        return _mupdf.PdfDocument_pdf_purge_locals_from_store(self)

    def pdf_read_journal(self, stm):
        """ Class-aware wrapper for `::pdf_read_journal()`."""
        return _mupdf.PdfDocument_pdf_read_journal(self, stm)

    def pdf_rearrange_pages(self, count, pages):
        """ Class-aware wrapper for `::pdf_rearrange_pages()`."""
        return _mupdf.PdfDocument_pdf_rearrange_pages(self, count, pages)

    def pdf_rearrange_pages2(self, pages):
        """ Class-aware wrapper for `::pdf_rearrange_pages2()`.   Swig-friendly wrapper for pdf_rearrange_pages()."""
        return _mupdf.PdfDocument_pdf_rearrange_pages2(self, pages)

    def pdf_redact_page(self, page, opts):
        """ Class-aware wrapper for `::pdf_redact_page()`."""
        return _mupdf.PdfDocument_pdf_redact_page(self, page, opts)

    def pdf_redo(self):
        """ Class-aware wrapper for `::pdf_redo()`."""
        return _mupdf.PdfDocument_pdf_redo(self)

    def pdf_repair_obj(self, buf, stmofsp, stmlenp, encrypt, id, page, tmpofs, root):
        """
        Class-aware wrapper for `::pdf_repair_obj()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_repair_obj(::pdf_lexbuf *buf, ::pdf_obj **encrypt, ::pdf_obj **id, ::pdf_obj **page, ::pdf_obj **root)` => `(int, int64_t stmofsp, int64_t stmlenp, int64_t tmpofs)`
        """
        return _mupdf.PdfDocument_pdf_repair_obj(self, buf, stmofsp, stmlenp, encrypt, id, page, tmpofs, root)

    def pdf_repair_obj_stms(self):
        """ Class-aware wrapper for `::pdf_repair_obj_stms()`."""
        return _mupdf.PdfDocument_pdf_repair_obj_stms(self)

    def pdf_repair_trailer(self):
        """ Class-aware wrapper for `::pdf_repair_trailer()`."""
        return _mupdf.PdfDocument_pdf_repair_trailer(self)

    def pdf_repair_xref(self):
        """ Class-aware wrapper for `::pdf_repair_xref()`."""
        return _mupdf.PdfDocument_pdf_repair_xref(self)

    def pdf_replace_xref(self, entries, n):
        """ Class-aware wrapper for `::pdf_replace_xref()`."""
        return _mupdf.PdfDocument_pdf_replace_xref(self, entries, n)

    def pdf_reset_form(self, fields, exclude):
        """ Class-aware wrapper for `::pdf_reset_form()`."""
        return _mupdf.PdfDocument_pdf_reset_form(self, fields, exclude)

    def pdf_resolve_link(self, uri, xp, yp):
        """
        Class-aware wrapper for `::pdf_resolve_link()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_resolve_link(const char *uri)` => `(int, float xp, float yp)`
        """
        return _mupdf.PdfDocument_pdf_resolve_link(self, uri, xp, yp)

    def pdf_rewrite_images(self, opts):
        """ Class-aware wrapper for `::pdf_rewrite_images()`."""
        return _mupdf.PdfDocument_pdf_rewrite_images(self, opts)

    def pdf_run_document_structure(self, dev, cookie):
        """ Class-aware wrapper for `::pdf_run_document_structure()`."""
        return _mupdf.PdfDocument_pdf_run_document_structure(self, dev, cookie)

    def pdf_run_glyph(self, resources, contents, dev, ctm, gstate, default_cs):
        """ Class-aware wrapper for `::pdf_run_glyph()`."""
        return _mupdf.PdfDocument_pdf_run_glyph(self, resources, contents, dev, ctm, gstate, default_cs)

    def pdf_save_document(self, filename, opts):
        """ Class-aware wrapper for `::pdf_save_document()`."""
        return _mupdf.PdfDocument_pdf_save_document(self, filename, opts)

    def pdf_save_journal(self, filename):
        """ Class-aware wrapper for `::pdf_save_journal()`."""
        return _mupdf.PdfDocument_pdf_save_journal(self, filename)

    def pdf_save_snapshot(self, filename):
        """ Class-aware wrapper for `::pdf_save_snapshot()`."""
        return _mupdf.PdfDocument_pdf_save_snapshot(self, filename)

    def pdf_select_layer_config(self, config_num):
        """ Class-aware wrapper for `::pdf_select_layer_config()`."""
        return _mupdf.PdfDocument_pdf_select_layer_config(self, config_num)

    def pdf_select_layer_config_ui(self, ui):
        """ Class-aware wrapper for `::pdf_select_layer_config_ui()`."""
        return _mupdf.PdfDocument_pdf_select_layer_config_ui(self, ui)

    def pdf_serialise_journal(self, out):
        """ Class-aware wrapper for `::pdf_serialise_journal()`."""
        return _mupdf.PdfDocument_pdf_serialise_journal(self, out)

    def pdf_set_annot_field_value(self, widget, text, ignore_trigger_events):
        """ Class-aware wrapper for `::pdf_set_annot_field_value()`."""
        return _mupdf.PdfDocument_pdf_set_annot_field_value(self, widget, text, ignore_trigger_events)

    def pdf_set_doc_event_callback(self, event_cb, free_event_data_cb, data):
        """ Class-aware wrapper for `::pdf_set_doc_event_callback()`."""
        return _mupdf.PdfDocument_pdf_set_doc_event_callback(self, event_cb, free_event_data_cb, data)

    def pdf_set_document_language(self, lang):
        """ Class-aware wrapper for `::pdf_set_document_language()`."""
        return _mupdf.PdfDocument_pdf_set_document_language(self, lang)

    def pdf_set_field_value(self, field, text, ignore_trigger_events):
        """ Class-aware wrapper for `::pdf_set_field_value()`."""
        return _mupdf.PdfDocument_pdf_set_field_value(self, field, text, ignore_trigger_events)

    def pdf_set_layer_config_as_default(self):
        """ Class-aware wrapper for `::pdf_set_layer_config_as_default()`."""
        return _mupdf.PdfDocument_pdf_set_layer_config_as_default(self)

    def pdf_set_page_labels(self, index, style, prefix, start):
        """ Class-aware wrapper for `::pdf_set_page_labels()`."""
        return _mupdf.PdfDocument_pdf_set_page_labels(self, index, style, prefix, start)

    def pdf_set_populating_xref_trailer(self, trailer):
        """ Class-aware wrapper for `::pdf_set_populating_xref_trailer()`."""
        return _mupdf.PdfDocument_pdf_set_populating_xref_trailer(self, trailer)

    def pdf_signature_byte_range(self, signature, byte_range):
        """ Class-aware wrapper for `::pdf_signature_byte_range()`."""
        return _mupdf.PdfDocument_pdf_signature_byte_range(self, signature, byte_range)

    def pdf_signature_contents(self, signature, contents):
        """
        Class-aware wrapper for `::pdf_signature_contents()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_signature_contents(::pdf_obj *signature)` => `(size_t, char *contents)`
        """
        return _mupdf.PdfDocument_pdf_signature_contents(self, signature, contents)

    def pdf_signature_hash_bytes(self, signature):
        """ Class-aware wrapper for `::pdf_signature_hash_bytes()`."""
        return _mupdf.PdfDocument_pdf_signature_hash_bytes(self, signature)

    def pdf_signature_incremental_change_since_signing(self, signature):
        """ Class-aware wrapper for `::pdf_signature_incremental_change_since_signing()`."""
        return _mupdf.PdfDocument_pdf_signature_incremental_change_since_signing(self, signature)

    def pdf_signature_is_signed(self, field):
        """ Class-aware wrapper for `::pdf_signature_is_signed()`."""
        return _mupdf.PdfDocument_pdf_signature_is_signed(self, field)

    def pdf_signature_set_value(self, field, signer, stime):
        """ Class-aware wrapper for `::pdf_signature_set_value()`."""
        return _mupdf.PdfDocument_pdf_signature_set_value(self, field, signer, stime)

    def pdf_subset_fonts(self, pages_len, pages):
        """ Class-aware wrapper for `::pdf_subset_fonts()`."""
        return _mupdf.PdfDocument_pdf_subset_fonts(self, pages_len, pages)

    def pdf_subset_fonts2(self, pages):
        """ Class-aware wrapper for `::pdf_subset_fonts2()`.   Swig-friendly wrapper for pdf_subset_fonts()."""
        return _mupdf.PdfDocument_pdf_subset_fonts2(self, pages)

    def pdf_toggle_layer_config_ui(self, ui):
        """ Class-aware wrapper for `::pdf_toggle_layer_config_ui()`."""
        return _mupdf.PdfDocument_pdf_toggle_layer_config_ui(self, ui)

    def pdf_trailer(self):
        """ Class-aware wrapper for `::pdf_trailer()`."""
        return _mupdf.PdfDocument_pdf_trailer(self)

    def pdf_undo(self):
        """ Class-aware wrapper for `::pdf_undo()`."""
        return _mupdf.PdfDocument_pdf_undo(self)

    def pdf_undoredo_state(self, steps):
        """
        Class-aware wrapper for `::pdf_undoredo_state()`.

        This method has out-params. Python/C# wrappers look like:
        	`pdf_undoredo_state()` => `(int, int steps)`
        """
        return _mupdf.PdfDocument_pdf_undoredo_state(self, steps)

    def pdf_undoredo_step(self, step):
        """ Class-aware wrapper for `::pdf_undoredo_step()`."""
        return _mupdf.PdfDocument_pdf_undoredo_step(self, step)

    def pdf_update_object(self, num, obj):
        """ Class-aware wrapper for `::pdf_update_object()`."""
        return _mupdf.PdfDocument_pdf_update_object(self, num, obj)

    def pdf_update_stream(self, ref, buf, compressed):
        """ Class-aware wrapper for `::pdf_update_stream()`."""
        return _mupdf.PdfDocument_pdf_update_stream(self, ref, buf, compressed)

    def pdf_update_xobject(self, xobj, bbox, mat, res, buffer):
        """ Class-aware wrapper for `::pdf_update_xobject()`."""
        return _mupdf.PdfDocument_pdf_update_xobject(self, xobj, bbox, mat, res, buffer)

    def pdf_validate_change_history(self):
        """ Class-aware wrapper for `::pdf_validate_change_history()`."""
        return _mupdf.PdfDocument_pdf_validate_change_history(self)

    def pdf_validate_changes(self, version):
        """ Class-aware wrapper for `::pdf_validate_changes()`."""
        return _mupdf.PdfDocument_pdf_validate_changes(self, version)

    def pdf_version(self):
        """ Class-aware wrapper for `::pdf_version()`."""
        return _mupdf.PdfDocument_pdf_version(self)

    def pdf_was_pure_xfa(self):
        """ Class-aware wrapper for `::pdf_was_pure_xfa()`."""
        return _mupdf.PdfDocument_pdf_was_pure_xfa(self)

    def pdf_was_repaired(self):
        """ Class-aware wrapper for `::pdf_was_repaired()`."""
        return _mupdf.PdfDocument_pdf_was_repaired(self)

    def pdf_write_document(self, out, opts):
        """ Class-aware wrapper for `::pdf_write_document()`."""
        return _mupdf.PdfDocument_pdf_write_document(self, out, opts)

    def pdf_write_journal(self, out):
        """ Class-aware wrapper for `::pdf_write_journal()`."""
        return _mupdf.PdfDocument_pdf_write_journal(self, out)

    def pdf_write_snapshot(self, out):
        """ Class-aware wrapper for `::pdf_write_snapshot()`."""
        return _mupdf.PdfDocument_pdf_write_snapshot(self, out)

    def pdf_xref_ensure_incremental_object(self, num):
        """ Class-aware wrapper for `::pdf_xref_ensure_incremental_object()`."""
        return _mupdf.PdfDocument_pdf_xref_ensure_incremental_object(self, num)

    def pdf_xref_ensure_local_object(self, num):
        """ Class-aware wrapper for `::pdf_xref_ensure_local_object()`."""
        return _mupdf.PdfDocument_pdf_xref_ensure_local_object(self, num)

    def pdf_xref_entry_map(self, fn, arg):
        """ Class-aware wrapper for `::pdf_xref_entry_map()`."""
        return _mupdf.PdfDocument_pdf_xref_entry_map(self, fn, arg)

    def pdf_xref_is_incremental(self, num):
        """ Class-aware wrapper for `::pdf_xref_is_incremental()`."""
        return _mupdf.PdfDocument_pdf_xref_is_incremental(self, num)

    def pdf_xref_len(self):
        """ Class-aware wrapper for `::pdf_xref_len()`."""
        return _mupdf.PdfDocument_pdf_xref_len(self)

    def pdf_xref_obj_is_unsaved_signature(self, obj):
        """ Class-aware wrapper for `::pdf_xref_obj_is_unsaved_signature()`."""
        return _mupdf.PdfDocument_pdf_xref_obj_is_unsaved_signature(self, obj)

    def pdf_xref_remove_unsaved_signature(self, field):
        """ Class-aware wrapper for `::pdf_xref_remove_unsaved_signature()`."""
        return _mupdf.PdfDocument_pdf_xref_remove_unsaved_signature(self, field)

    def pdf_xref_store_unsaved_signature(self, field, signer):
        """ Class-aware wrapper for `::pdf_xref_store_unsaved_signature()`."""
        return _mupdf.PdfDocument_pdf_xref_store_unsaved_signature(self, field, signer)

    def super(self):
        """ Returns wrapper for .super member."""
        return _mupdf.PdfDocument_super(self)

    def __init__(self, *args):
        """
        *Overload 1:*
        == Constructors.  Constructor using `pdf_create_document()`.

        |

        *Overload 2:*
        Constructor using `pdf_document_from_fz_document()`.

        |

        *Overload 3:*
        Constructor using `pdf_open_document()`.

        |

        *Overload 4:*
        Constructor using `pdf_open_document_with_stream()`.

        |

        *Overload 5:*
        Copy constructor using `pdf_keep_document()`.

        |

        *Overload 6:*
        Constructor using raw copy of pre-existing `::pdf_document`.
        """
        _mupdf.PdfDocument_swiginit(self, _mupdf.new_PdfDocument(*args))
    __swig_destroy__ = _mupdf.delete_PdfDocument

    def m_internal_value(self):
        """ Return numerical value of .m_internal; helps with Python debugging."""
        return _mupdf.PdfDocument_m_internal_value(self)
    m_internal = property(_mupdf.PdfDocument_m_internal_get, _mupdf.PdfDocument_m_internal_set, doc=' Pointer to wrapped data.')
    s_num_instances = property(_mupdf.PdfDocument_s_num_instances_get, _mupdf.PdfDocument_s_num_instances_set)