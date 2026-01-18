from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Any
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
import six
def _GetSubElement(self, top_element, path):
    parts = path.split('.')[1:]
    current = top_element
    for part in parts:
        current = current.LoadSubElement(part)
        if not current:
            return None
    return current