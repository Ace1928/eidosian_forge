import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _string_split_transformer(parent, node, full_name, name, logs):
    """Update tf.string_split arguments: skip_empty, sep, result_type, source."""
    for i, kw in enumerate(node.keywords):
        if kw.arg == 'skip_empty':
            if _is_ast_false(kw.value):
                logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'removed argument skip_empty for tf.string_split.'))
                node.keywords.pop(i)
                break
            else:
                return _rename_to_compat_v1(node, full_name, logs, "tf.string_split's replacement no longer takes the skip_empty argument.")
    found_sep = False
    for i, kw in enumerate(node.keywords):
        if kw.arg == 'sep':
            found_sep = True
            if isinstance(kw.value, ast.Str):
                if kw.value.s == '':
                    node = _rename_func(node, full_name, 'tf.strings.bytes_split', logs, 'Splitting bytes is not handled by tf.strings.bytes_split().')
                    node.keywords.pop(i)
            else:
                return _rename_to_compat_v1(node, full_name, logs, "The semantics for tf.string_split's sep parameter have changed when sep is the empty string; but sep is not a string literal, so we can't tell if it's an empty string.")
    if not found_sep:
        return _rename_to_compat_v1(node, full_name, logs, "The semantics for tf.string_split's sep parameter have changed when sep unspecified: it now splits on all whitespace, not just the space character.")
    return _string_split_rtype_transformer(parent, node, full_name, name, logs)