from collections import defaultdict
from inspect import Parameter
from jedi import debug
from jedi.inference.utils import PushBackIterator
from jedi.inference import analysis
from jedi.inference.lazy_value import LazyKnownValue, \
from jedi.inference.value import iterable
from jedi.inference.names import ParamName
def _error_argument_count(funcdef, actual_count):
    params = funcdef.get_params()
    default_arguments = sum((1 for p in params if p.default or p.star_count))
    if default_arguments == 0:
        before = 'exactly '
    else:
        before = 'from %s to ' % (len(params) - default_arguments)
    return 'TypeError: %s() takes %s%s arguments (%s given).' % (funcdef.name, before, len(params), actual_count)