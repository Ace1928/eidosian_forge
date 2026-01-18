from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core.cache import resource_cache
from googlecloudsdk.core.console import console_attr
import six
def _CompleteFromGenericCompleterClass(self, prefix=''):
    """Helper to complete from a class that isn't a cache completer."""
    completer = None
    try:
        completer = self._completer_class()
        return completer(prefix=prefix)
    except BaseException as e:
        return self._HandleCompleterException(e, prefix=prefix, completer=completer)