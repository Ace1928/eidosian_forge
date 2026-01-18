from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import batch_helper
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import single_request_helper
from googlecloudsdk.api_lib.util import exceptions as http_exceptions
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def WaitForOperations(operations_data, http, batch_url, warnings, errors, progress_tracker=None, timeout=None, log_result=True):
    """Blocks until the given operations are done or until a timeout is reached.

  Args:
    operations_data: A list of OperationData objects holding Operations to poll.
    http: An HTTP object.
    batch_url: The URL to which batch requests should be sent.
    warnings: An output parameter for capturing warnings.
    errors: An output parameter for capturing errors.
    progress_tracker: progress tracker to tick while waiting for operations to
                      finish.
    timeout: The maximum amount of time, in seconds, to wait for the
      operations to reach the DONE state.
    log_result: Whether the Operation Waiter should print the result in past
      tense of each request.

  Yields:
    The resources pointed to by the operations' targetLink fields if
    the operation type is not delete. Only resources whose
    corresponding operations reach done are yielded.
  """
    if not operations_data:
        return
    timeout = timeout or _POLLING_TIMEOUT_SEC
    operation_details = {}
    unprocessed_operations = []
    for operation in operations_data:
        operation_details[operation.operation.selfLink] = operation
        unprocessed_operations.append((operation.operation, _SERVICE_UNAVAILABLE_RETRY_COUNT))
    start = time_util.CurrentTimeSec()
    sleep_sec = 0
    operation_type = operations_data[0].operation_service.GetResponseType('Get')
    while unprocessed_operations:
        if progress_tracker:
            progress_tracker.Tick()
        resource_requests = []
        operation_requests = []
        log.debug('Operations to inspect: %s', unprocessed_operations)
        for operation, _ in unprocessed_operations:
            data = operation_details[operation.selfLink]
            data.SetOperation(operation)
            operation_service = data.operation_service
            resource_service = data.resource_service
            if operation.status == operation_type.StatusValueValuesEnum.DONE:
                _RecordProblems(operation, warnings, errors)
                if operation.httpErrorStatusCode and operation.httpErrorStatusCode != 200:
                    if data.always_return_operation:
                        yield operation
                    else:
                        continue
                if operation.error:
                    continue
                if data.no_followup:
                    yield operation
                    continue
                if not _IsDeleteOp(operation.operationType):
                    request = data.ResourceGetRequest()
                    if request:
                        resource_requests.append((resource_service, 'Get', request))
                if operation.targetLink and log_result:
                    log.status.write('{0} [{1}].\n'.format(_HumanFriendlyNameForOpPastTense(operation.operationType).capitalize(), operation.targetLink))
            elif data.IsGlobalOrganizationOperation():
                request = data.OperationGetRequest()
                operation_requests.append((operation_service, 'Get', request))
            else:
                request = data.OperationWaitRequest()
                operation_requests.append((operation_service, 'Wait', request))
        requests = resource_requests + operation_requests
        if not requests:
            break
        if not properties.VALUES.compute.force_batch_request.GetBool() and len(requests) == 1:
            service, method, request_body = requests[0]
            responses, request_errors = single_request_helper.MakeSingleRequest(service=service, method=method, request_body=request_body)
        else:
            responses, request_errors = batch_helper.MakeRequests(requests=requests, http=http, batch_url=batch_url)
        all_done = True
        previous_operations = unprocessed_operations
        current_errors = list(request_errors)
        unprocessed_operations = []
        for seq, response in enumerate(responses):
            if isinstance(response, operation_type):
                unprocessed_operations.append((response, _SERVICE_UNAVAILABLE_RETRY_COUNT))
                if response.status != operation_type.StatusValueValuesEnum.DONE:
                    all_done = False
            elif response is None and request_errors and (request_errors[0][0] == 503):
                error = request_errors.pop(0)
                operation, retry_count = previous_operations[seq]
                retry_count -= 1
                if retry_count > 0:
                    unprocessed_operations.append((operation, retry_count))
                    all_done = False
                else:
                    errors.append(error)
            else:
                yield response
        errors.extend(request_errors)
        if not unprocessed_operations:
            break
        if all_done:
            continue
        if time_util.CurrentTimeSec() - start > timeout:
            if not current_errors:
                log.debug('Timeout of %ss reached.', timeout)
                _RecordUnfinishedOperations(unprocessed_operations, errors)
            else:
                errors.extend(current_errors)
            break
        sleep_sec = min(sleep_sec + 1, _MAX_TIME_BETWEEN_POLLS_SEC)
        log.debug('Sleeping for %ss.', sleep_sec)
        time_util.Sleep(sleep_sec)