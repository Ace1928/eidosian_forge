import collections
import functools
import inspect
import re
from tensorflow.python.framework import strict_mode
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.docs import doc_controls
def _validate_deprecation_args(date, instructions):
    if date is not None and (not re.match('20\\d\\d-[01]\\d-[0123]\\d', date)):
        raise ValueError(f'Date must be in format YYYY-MM-DD. Received: {date}')
    if not instructions:
        raise ValueError("Don't deprecate things without conversion instructions! Specify the `instructions` argument.")