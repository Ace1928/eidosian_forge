from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class WorkerConfigButNoWorkerpoolError(exceptions.Error):
    """The user has not supplied a worker pool even though a workerconfig has been specified."""

    def __init__(self):
        msg = 'Detected a worker pool config but no worker pool. Please specify a worker pool.'
        super(WorkerConfigButNoWorkerpoolError, self).__init__(msg)