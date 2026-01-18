import ast
import inspect
import io
import linecache
import re
import sys
import textwrap
import tokenize
import astunparse
import gast
from tensorflow.python.autograph.pyct import errors
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.util import tf_inspect
def _parse_lambda(lam):
    """Returns the AST and source code of given lambda function.

  Args:
    lam: types.LambdaType, Python function/method/class

  Returns:
    gast.AST, Text: the parsed AST node; the source code that was parsed to
    generate the AST (including any prefixes that this function may have added).
  """
    mod = inspect.getmodule(lam)
    f = inspect.getsourcefile(lam)
    def_line = lam.__code__.co_firstlineno
    lines = linecache.getlines(f, mod.__dict__)
    source = ''.join(lines)
    all_nodes = parse(source, preamble_len=0, single_node=False)
    search_nodes = []
    for node in all_nodes:
        if getattr(node, 'lineno', def_line) <= def_line:
            search_nodes.append(node)
        else:
            break
    lambda_nodes = []
    for node in search_nodes:
        lambda_nodes.extend((n for n in gast.walk(node) if isinstance(n, gast.Lambda)))
    candidates = []
    for ln in lambda_nodes:
        minl, maxl = (MAX_SIZE, 0)
        for n in gast.walk(ln):
            minl = min(minl, getattr(n, 'lineno', minl))
            lineno = getattr(n, 'lineno', maxl)
            end_lineno = getattr(n, 'end_lineno', None)
            if end_lineno is not None:
                lineno = end_lineno
            maxl = max(maxl, lineno)
        if minl <= def_line <= maxl:
            candidates.append((ln, minl, maxl))
    if len(candidates) == 1:
        (node, minl, maxl), = candidates
        return _without_context(node, lines, minl, maxl)
    elif not candidates:
        lambda_codes = '\n'.join([unparse(l) for l in lambda_nodes])
        raise errors.UnsupportedLanguageElementError(f'could not parse the source code of {lam}: no matching AST found among candidates:\n{lambda_codes}')
    matches = [v for v in candidates if _node_matches_argspec(v[0], lam)]
    if len(matches) == 1:
        (node, minl, maxl), = matches
        return _without_context(node, lines, minl, maxl)
    matches = '\n'.join(('Match {}:\n{}\n'.format(i, unparse(node, include_encoding_marker=False)) for i, (node, _, _) in enumerate(matches)))
    raise errors.UnsupportedLanguageElementError(f'could not parse the source code of {lam}: found multiple definitions with identical signatures at the location. This error may be avoided by defining each lambda on a single line and with unique argument names. The matching definitions were:\n{matches}')