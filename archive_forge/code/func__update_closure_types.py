import itertools
from typing import Any, Callable, Dict, Set
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import annos
def _update_closure_types(self, ast_node, types):
    existing_types = anno.Static.CLOSURE_TYPES.of(ast_node, None)
    if existing_types is None:
        existing_types = {}
        anno.Static.CLOSURE_TYPES.add_to(ast_node, existing_types)
    for k, v in types.types.items():
        if k in existing_types:
            existing_types[k].update(v)
        else:
            existing_types[k] = set(v)