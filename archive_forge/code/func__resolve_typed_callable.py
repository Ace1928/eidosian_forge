import itertools
from typing import Any, Callable, Dict, Set
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import annos
def _resolve_typed_callable(self, f_types, arg_types, keyword_types):
    ret_types = set()
    for t in f_types:
        if isinstance(t, Callable):
            args = t.__args__
            if args:
                ret_types.add(args[-1])
            else:
                ret_types.add(Any)
        else:
            raise NotImplementedError('callable type {}'.format(type(t)))
    side_effects = None
    return (ret_types, side_effects)