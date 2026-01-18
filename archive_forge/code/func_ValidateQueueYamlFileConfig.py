from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.tasks import task_queues_convertors as convertors
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.tasks import flags
from googlecloudsdk.command_lib.tasks import parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
import six
from six.moves import urllib
def ValidateQueueYamlFileConfig(config):
    """Validates queue configuration parameters in the queue YAML file.

  The purpose of this function is to mimick the behaviour of the old
  implementation of `gcloud app deploy queue.yaml` before migrating away
  from console-admin-hr. The errors generated are the same as the ones
  previously seen when gcloud sent the batch-request for updating queues to the
  Zeus backend.

  Args:
     config: A yaml_parsing.ConfigYamlInfo object for the parsed YAML file we
      are going to process.

  Raises:
    HTTPError: Various different scenarios defined in the function can cause
      this exception to be raised.
  """
    queue_yaml = config.parsed
    if not queue_yaml.queue:
        return
    for queue in queue_yaml.queue:
        if not queue.mode or queue.mode == constants.PUSH_QUEUE:
            if not queue.rate:
                _RaiseHTTPException('Invalid queue configuration. Refill rate must be specified for push-based queue.')
            else:
                rate_in_seconds = convertors.ConvertRate(queue.rate)
                if rate_in_seconds > constants.MAX_RATE:
                    _RaiseHTTPException('Invalid queue configuration. Refill rate must not exceed {} per second (is {:.1f}).'.format(constants.MAX_RATE, rate_in_seconds))
            if queue.retry_parameters:
                _ValidateTaskRetryLimit(queue)
                if queue.retry_parameters.task_age_limit and int(convertors.CheckAndConvertStringToFloatIfApplicable(queue.retry_parameters.task_age_limit)) <= 0:
                    _RaiseHTTPException('Invalid queue configuration. Task age limit must be greater than zero.')
                if queue.retry_parameters.min_backoff_seconds and queue.retry_parameters.min_backoff_seconds < 0:
                    _RaiseHTTPException('Invalid queue configuration. Min backoff seconds must not be less than zero.')
                if queue.retry_parameters.max_backoff_seconds and queue.retry_parameters.max_backoff_seconds < 0:
                    _RaiseHTTPException('Invalid queue configuration. Max backoff seconds must not be less than zero.')
                if queue.retry_parameters.max_doublings and queue.retry_parameters.max_doublings < 0:
                    _RaiseHTTPException('Invalid queue configuration. Max doublings must not be less than zero.')
                if queue.retry_parameters.min_backoff_seconds is not None and queue.retry_parameters.max_backoff_seconds is not None:
                    min_backoff = queue.retry_parameters.min_backoff_seconds
                    max_backoff = queue.retry_parameters.max_backoff_seconds
                    if max_backoff < min_backoff:
                        _RaiseHTTPException('Invalid queue configuration. Min backoff sec must not be greater than than max backoff sec.')
            if queue.bucket_size:
                if queue.bucket_size < 0:
                    _RaiseHTTPException('Error updating queue "{}": The queue rate is invalid.'.format(queue.name))
                elif queue.bucket_size > constants.MAX_BUCKET_SIZE:
                    _RaiseHTTPException('Error updating queue "{}": Maximum bucket size is {}.'.format(queue.name, constants.MAX_BUCKET_SIZE))
        else:
            if queue.rate:
                _RaiseHTTPException('Invalid queue configuration. Refill rate must not be specified for pull-based queue.')
            if queue.retry_parameters:
                _ValidateTaskRetryLimit(queue)
                if queue.retry_parameters.task_age_limit is not None:
                    _RaiseHTTPException("Invalid queue configuration. Can't specify task_age_limit for a pull queue.")
                if queue.retry_parameters.min_backoff_seconds is not None:
                    _RaiseHTTPException("Invalid queue configuration. Can't specify min_backoff_seconds for a pull queue.")
                if queue.retry_parameters.max_backoff_seconds is not None:
                    _RaiseHTTPException("Invalid queue configuration. Can't specify max_backoff_seconds for a pull queue.")
                if queue.retry_parameters.max_doublings is not None:
                    _RaiseHTTPException("Invalid queue configuration. Can't specify max_doublings for a pull queue.")
            if queue.max_concurrent_requests is not None:
                _RaiseHTTPException('Invalid queue configuration. Max concurrent requests must not be specified for pull-based queue.')
            if queue.bucket_size is not None:
                _RaiseHTTPException('Invalid queue configuration. Bucket size must not be specified for pull-based queue.')
            if queue.target:
                _RaiseHTTPException('Invalid queue configuration. Target must not be specified for pull-based queue.')