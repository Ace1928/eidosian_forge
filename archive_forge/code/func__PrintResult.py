from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import log
from googlecloudsdk.core.console import progress_tracker
def _PrintResult(self, result):
    self._Print(result.message, not result.passed)