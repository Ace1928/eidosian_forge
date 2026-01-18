from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _validate_option(self, option, val) -> None:
    if option == 'field_names':
        self._validate_field_names(val)
    elif option == 'none_format':
        self._validate_none_format(val)
    elif option in ('start', 'end', 'max_width', 'min_width', 'min_table_width', 'max_table_width', 'padding_width', 'left_padding_width', 'right_padding_width', 'format'):
        self._validate_nonnegative_int(option, val)
    elif option == 'sortby':
        self._validate_field_name(option, val)
    elif option == 'sort_key':
        self._validate_function(option, val)
    elif option == 'hrules':
        self._validate_hrules(option, val)
    elif option == 'vrules':
        self._validate_vrules(option, val)
    elif option == 'fields':
        self._validate_all_field_names(option, val)
    elif option in ('header', 'border', 'preserve_internal_border', 'reversesort', 'xhtml', 'print_empty', 'oldsortslice'):
        self._validate_true_or_false(option, val)
    elif option == 'header_style':
        self._validate_header_style(val)
    elif option == 'int_format':
        self._validate_int_format(option, val)
    elif option == 'float_format':
        self._validate_float_format(option, val)
    elif option == 'custom_format':
        for k, formatter in val.items():
            self._validate_function(f'{option}.{k}', formatter)
    elif option in ('vertical_char', 'horizontal_char', 'horizontal_align_char', 'junction_char', 'top_junction_char', 'bottom_junction_char', 'right_junction_char', 'left_junction_char', 'top_right_junction_char', 'top_left_junction_char', 'bottom_right_junction_char', 'bottom_left_junction_char'):
        self._validate_single_char(option, val)
    elif option == 'attributes':
        self._validate_attributes(option, val)