import copy
import re
import numpy as np
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.lib import debug_data
def _counts_summary(counts, skip_zeros=True, total_count=None):
    """Format values as a two-row table."""
    if skip_zeros:
        counts = [(count_key, count_val) for count_key, count_val in counts if count_val]
    max_common_len = 0
    for count_key, count_val in counts:
        count_val_str = str(count_val)
        common_len = max(len(count_key) + 1, len(count_val_str) + 1)
        max_common_len = max(common_len, max_common_len)
    key_line = debugger_cli_common.RichLine('|')
    val_line = debugger_cli_common.RichLine('|')
    for count_key, count_val in counts:
        count_val_str = str(count_val)
        key_line += _pad_string_to_length(count_key, max_common_len)
        val_line += _pad_string_to_length(count_val_str, max_common_len)
    key_line += ' |'
    val_line += ' |'
    if total_count is not None:
        total_key_str = 'total'
        total_val_str = str(total_count)
        max_common_len = max(len(total_key_str) + 1, len(total_val_str))
        total_key_str = _pad_string_to_length(total_key_str, max_common_len)
        total_val_str = _pad_string_to_length(total_val_str, max_common_len)
        key_line += total_key_str + ' |'
        val_line += total_val_str + ' |'
    return debugger_cli_common.rich_text_lines_from_rich_line_list([key_line, val_line])