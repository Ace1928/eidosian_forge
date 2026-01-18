import functools
import importlib
import inspect
import os
import sys
import textwrap
import traceback
from tensorflow.python.autograph import operators
from tensorflow.python.autograph import utils
from tensorflow.python.autograph.converters import asserts
from tensorflow.python.autograph.converters import break_statements
from tensorflow.python.autograph.converters import call_trees
from tensorflow.python.autograph.converters import conditional_expressions
from tensorflow.python.autograph.converters import continue_statements
from tensorflow.python.autograph.converters import control_flow
from tensorflow.python.autograph.converters import directives
from tensorflow.python.autograph.converters import functions
from tensorflow.python.autograph.converters import lists
from tensorflow.python.autograph.converters import logical_expressions
from tensorflow.python.autograph.converters import return_statements
from tensorflow.python.autograph.converters import slices
from tensorflow.python.autograph.converters import variables
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.core import function_wrappers
from tensorflow.python.autograph.core import unsupported_features_checker
from tensorflow.python.autograph.impl import conversion
from tensorflow.python.autograph.lang import special_functions
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import error_utils
from tensorflow.python.autograph.pyct import errors
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transpiler
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.autograph.utils import ag_logging as logging
from tensorflow.python.eager.polymorphic_function import tf_method_target
from tensorflow.python.framework import errors_impl
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import tf_stack
from tensorflow.python.util.tf_export import tf_export
def _fall_back_unconverted(f, args, kwargs, options, exc):
    """Falls back to calling the function unconverted, in case of error."""
    warning_template = 'AutoGraph could not transform %s and will run it as-is.\n%sCause: %s\nTo silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert'
    if isinstance(exc, errors.InaccessibleSourceCodeError):
        if ag_ctx.INSPECT_SOURCE_SUPPORTED:
            logging.warning(warning_template, f, '', exc)
    elif isinstance(exc, errors.UnsupportedLanguageElementError):
        if not conversion.is_in_allowlist_cache(f, options):
            logging.warning(warning_template, f, '', exc)
    else:
        file_bug_message = 'Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.\n'
        logging.warning(warning_template, f, file_bug_message, exc)
    return _call_unconverted(f, args, kwargs, options)