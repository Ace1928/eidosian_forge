import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _replace_mode(parent, old_value):
    """Replaces old_value with (old_value).lower()."""
    new_value = pasta.parse('mode.lower()')
    mode = new_value.body[0].value.func
    pasta.ast_utils.replace_child(mode, mode.value, old_value)
    pasta.ast_utils.replace_child(parent, old_value, new_value)
    pasta.base.formatting.set(old_value, 'prefix', '(')
    pasta.base.formatting.set(old_value, 'suffix', ')')