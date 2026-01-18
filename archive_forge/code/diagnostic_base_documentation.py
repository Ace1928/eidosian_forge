from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import log
from googlecloudsdk.core.console import progress_tracker
Runs one or more checks, tries fixes, and outputs results.

    Returns:
      True if the diagnostic ultimately passed.
    