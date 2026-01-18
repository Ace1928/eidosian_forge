from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import signal
from googlecloudsdk.api_lib.firebase.test import exit_code
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def _Handler(self, unused_signal, unused_frame):
    log.status.write('\n\nCancelling test [{id}]...\n\n'.format(id=self._matrix_monitor.matrix_id))
    self._matrix_monitor.CancelTestMatrix()
    log.status.write('\nTest matrix has been cancelled.\n')
    raise exceptions.ExitCodeNoError(exit_code=exit_code.MATRIX_CANCELLED)