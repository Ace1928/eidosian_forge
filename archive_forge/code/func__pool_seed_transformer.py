import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2
def _pool_seed_transformer(parent, node, full_name, name, logs):
    """Removes seed2 and deterministic, and adds non-zero seed if needed."""
    seed_arg = None
    deterministic = False
    modified = False
    new_keywords = []
    for kw in node.keywords:
        if sys.version_info[:2] >= (3, 5) and isinstance(kw, ast.Starred):
            pass
        elif kw.arg == 'seed':
            seed_arg = kw
        elif kw.arg == 'seed2' or kw.arg == 'deterministic':
            lineno = getattr(kw, 'lineno', node.lineno)
            col_offset = getattr(kw, 'col_offset', node.col_offset)
            logs.append((ast_edits.INFO, lineno, col_offset, 'Removed argument %s for function %s' % (kw.arg, full_name or name)))
            if kw.arg == 'deterministic':
                if not _is_ast_false(kw.value):
                    deterministic = True
            modified = True
            continue
        new_keywords.append(kw)
    if deterministic:
        if seed_arg is None:
            new_keywords.append(ast.keyword(arg='seed', value=ast.Num(42)))
            logs.add((ast_edits.INFO, node.lineno, node.col_offset, 'Adding seed=42 to call to %s since determinism was requested' % (full_name or name)))
        else:
            logs.add((ast_edits.WARNING, node.lineno, node.col_offset, 'The deterministic argument is deprecated for %s, pass a non-zero seed for determinism. The deterministic argument is present, possibly not False, and the seed is already set. The converter cannot determine whether it is nonzero, please check.'))
    if modified:
        node.keywords = new_keywords
        return node
    else:
        return