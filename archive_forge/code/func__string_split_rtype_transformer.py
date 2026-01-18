import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _string_split_rtype_transformer(parent, node, full_name, name, logs):
    """Update tf.strings.split arguments: result_type, source."""
    need_to_sparse = True
    for i, kw in enumerate(node.keywords):
        if kw.arg == 'result_type':
            if isinstance(kw.value, ast.Str) and kw.value.s in ('RaggedTensor', 'SparseTensor'):
                logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Removed argument result_type=%r for function %s' % (kw.value.s, full_name or name)))
                node.keywords.pop(i)
                if kw.value.s == 'RaggedTensor':
                    need_to_sparse = False
            else:
                return _rename_to_compat_v1(node, full_name, logs, '%s no longer takes the result_type parameter.' % full_name)
            break
    for i, kw in enumerate(node.keywords):
        if kw.arg == 'source':
            kw.arg = 'input'
    if need_to_sparse:
        if isinstance(parent, ast.Attribute) and parent.attr == 'to_sparse':
            return
        logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Adding call to RaggedTensor.to_sparse() to result of strings.split, since it now returns a RaggedTensor.'))
        node = ast.Attribute(value=copy.deepcopy(node), attr='to_sparse')
        try:
            node = ast.Call(node, [], [])
        except TypeError:
            node = ast.Call(node, [], [], None, None)
    return node