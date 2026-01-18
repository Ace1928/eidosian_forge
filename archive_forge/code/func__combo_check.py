from functools import partial
from itertools import product
from .core import make_vjp, make_jvp, vspace
from .util import subvals
from .wrap_util import unary_to_nary, get_name
def _combo_check(*args, **kwargs):
    kwarg_key_vals = [[(k, x) for x in xs] for k, xs in kwargs.items()]
    for _args in product(*args):
        for _kwargs in product(*kwarg_key_vals):
            _check_grads(fun)(*_args, **dict(_kwargs))