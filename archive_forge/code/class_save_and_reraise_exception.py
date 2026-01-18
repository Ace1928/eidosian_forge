import functools
import io
import logging
import os
import sys
import time
import traceback
from oslo_utils import encodeutils
from oslo_utils import reflection
from oslo_utils import timeutils
class save_and_reraise_exception(object):
    """Save current exception, run some code and then re-raise.

    In some cases the exception context can be cleared, resulting in None
    being attempted to be re-raised after an exception handler is run. This
    can happen when eventlet switches greenthreads or when running an
    exception handler, code raises and catches an exception. In both
    cases the exception context will be cleared.

    To work around this, we save the exception state, run handler code, and
    then re-raise the original exception. If another exception occurs, the
    saved exception is logged and the new exception is re-raised.

    In some cases the caller may not want to re-raise the exception, and
    for those circumstances this context provides a reraise flag that
    can be used to suppress the exception.  For example::

      except Exception:
          with save_and_reraise_exception() as ctxt:
              decide_if_need_reraise()
              if not should_be_reraised:
                  ctxt.reraise = False

    If another exception occurs and reraise flag is False,
    the saved exception will not be logged.

    If the caller wants to raise new exception during exception handling
    he/she sets reraise to False initially with an ability to set it back to
    True if needed::

      except Exception:
          with save_and_reraise_exception(reraise=False) as ctxt:
              [if statements to determine whether to raise a new exception]
              # Not raising a new exception, so reraise
              ctxt.reraise = True

    .. versionchanged:: 1.4
       Added *logger* optional parameter.
    """

    def __init__(self, reraise=True, logger=None):
        self.reraise = reraise
        if logger is None:
            logger = logging.getLogger()
        self.logger = logger
        self.type_, self.value, self.tb = (None, None, None)

    def force_reraise(self):
        if self.type_ is None and self.value is None:
            raise RuntimeError('There is no (currently) captured exception to force the reraising of')
        try:
            if self.value is None:
                self.value = self.type_()
            if self.value.__traceback__ is not self.tb:
                raise self.value.with_traceback(self.tb)
            raise self.value
        finally:
            self.value = None
            self.tb = None

    def capture(self, check=True):
        type_, value, tb = sys.exc_info()
        if check and type_ is None and (value is None):
            raise RuntimeError('There is no active exception to capture')
        self.type_, self.value, self.tb = (type_, value, tb)
        return self

    def __enter__(self):
        return self.capture(check=False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if self.reraise:
                self.logger.error('Original exception being dropped: %s', traceback.format_exception(self.type_, self.value, self.tb))
            return False
        if self.reraise:
            self.force_reraise()