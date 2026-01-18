from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core.cache import resource_cache
from googlecloudsdk.core.console import console_attr
import six
def _CompleteFromFunction(self, prefix=''):
    """Helper to complete from a function completer."""
    try:
        return self._completer_class(prefix)
    except BaseException as e:
        return self._HandleCompleterException(e, prefix=prefix)