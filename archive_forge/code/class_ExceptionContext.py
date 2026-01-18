from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
import traceback
from googlecloudsdk.core.util import encoding
import six
class ExceptionContext(object):
    """An exception context that can be re-raised outside of try-except.

  Usage:
    exception_context = None
    ...
    try:
      ...
    except ... e:
      # This MUST be called in the except: clause.
      exception_context = exceptions.ExceptionContext(e)
    ...
    if exception_context:
      exception_context.Reraise()
  """

    def __init__(self, e):
        self._exception = e
        self._traceback = sys.exc_info()[2]
        if not self._traceback:
            raise InternalError('Must set ExceptionContext within an except clause.')

    def Reraise(self):
        six.reraise(type(self._exception), self._exception, self._traceback)