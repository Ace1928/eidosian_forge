from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import time
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import logs as cloudbuild_logs
from googlecloudsdk.api_lib.util import requests
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from six.moves import range  # pylint: disable=redefined-builtin
def WaitAndStreamLogs(self, build_op):
    """Wait for a Cloud Build to finish, streaming logs if possible."""
    build_id = GetBuildProp(build_op, 'id', required=True)
    logs_uri = GetBuildProp(build_op, 'logUrl')
    logs_bucket = GetBuildProp(build_op, 'logsBucket')
    log.status.Print('Started cloud build [{build_id}].'.format(build_id=build_id))
    log_loc = 'in the Cloud Console.'
    log_tailer = None
    if logs_bucket:
        log_object = self.CLOUDBUILD_LOGFILE_FMT_STRING.format(build_id=build_id)
        log_tailer = cloudbuild_logs.GCSLogTailer(bucket=logs_bucket, obj=log_object)
        if logs_uri:
            log.status.Print('To see logs in the Cloud Console: ' + logs_uri)
            log_loc = 'at ' + logs_uri
        else:
            log.status.Print('Logs can be found in the Cloud Console.')
    callback = None
    if log_tailer:
        callback = log_tailer.Poll
    try:
        op = self.WaitForOperation(operation=build_op, retry_callback=callback)
    except OperationTimeoutError:
        log.debug('', exc_info=True)
        raise BuildFailedError('Cloud build timed out. Check logs ' + log_loc)
    if log_tailer:
        log_tailer.Poll(is_last=True)
    final_status = _GetStatusFromOp(op)
    if final_status != self.CLOUDBUILD_SUCCESS:
        message = requests.ExtractErrorMessage(encoding.MessageToPyValue(op.error))
        raise BuildFailedError('Cloud build failed. Check logs ' + log_loc + ' Failure status: ' + final_status + ': ' + message)