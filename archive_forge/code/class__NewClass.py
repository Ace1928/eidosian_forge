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
class _NewClass(func_or_class):
    __doc__ = decorator_utils.add_notice_to_docstring(func_or_class.__doc__, 'Please use %s instead.' % name, 'DEPRECATED CLASS', '(deprecated)', ['THIS CLASS IS DEPRECATED. It will be removed in a future version. '], notice_type='Deprecated')
    __name__ = func_or_class.__name__
    __module__ = _call_location(outer=True)

    @_wrap_decorator(func_or_class.__init__, 'deprecated_alias')
    def __init__(self, *args, **kwargs):
        if hasattr(_NewClass.__init__, '__func__'):
            _NewClass.__init__.__func__.__doc__ = func_or_class.__init__.__doc__
        else:
            _NewClass.__init__.__doc__ = func_or_class.__init__.__doc__
        if _PRINT_DEPRECATION_WARNINGS:
            if _NewClass.__init__ not in _PRINTED_WARNING:
                if warn_once:
                    _PRINTED_WARNING[_NewClass.__init__] = True
                _log_deprecation('From %s: The name %s is deprecated. Please use %s instead.\n', _call_location(), deprecated_name, name)
        super(_NewClass, self).__init__(*args, **kwargs)