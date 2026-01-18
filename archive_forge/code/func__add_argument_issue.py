from collections import defaultdict
from inspect import Parameter
from jedi import debug
from jedi.inference.utils import PushBackIterator
from jedi.inference import analysis
from jedi.inference.lazy_value import LazyKnownValue, \
from jedi.inference.value import iterable
from jedi.inference.names import ParamName
def _add_argument_issue(error_name, lazy_value, message):
    if isinstance(lazy_value, LazyTreeValue):
        node = lazy_value.data
        if node.parent.type == 'argument':
            node = node.parent
        return analysis.add(lazy_value.context, error_name, node, message)