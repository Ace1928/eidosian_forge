import sys
import typing
from rpy2.rinterface_lib import embedded
from rpy2.rinterface_lib import callbacks
from rpy2.rinterface_lib import openrlib
Initialize the embedded R.

    :param interactive: Should R run in interactive or non-interactive mode?
    if `None` the value in `_DEFAULT_R_INTERACTIVE` will be used.
    :param _want_setcallbacks: Should custom rpy2 callbacks for R frontends
    be set?.
    :param _c_stack_limit: Limit for the C Stack.
    if `None` the value in `_DEFAULT_C_STACK_LIMIT` will be used.
    