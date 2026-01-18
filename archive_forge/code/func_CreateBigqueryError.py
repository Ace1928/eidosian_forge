import textwrap
from typing import Dict, List, Optional
import bq_flags
from utils import bq_logging
def CreateBigqueryError(error: Dict[str, str], server_error: Dict[str, str], error_ls: List[Dict[str, str]], job_ref: Optional[str]=None, session_id: Optional[str]=None) -> BigqueryError:
    """Returns a BigqueryError for json error embedded in server_error.

  If error_ls contains any errors other than the given one, those
  are also included in the returned message.

  Args:
    error: The primary error to convert.
    server_error: The error returned by the server. (This is only used in the
      case that error is malformed.)
    error_ls: Additional errors to include in the error message.
    job_ref: String representation a JobReference, if this is an error
      associated with a job.
    session_id: Id of the session if the job is part of one.

  Returns:
    BigqueryError representing error.
  """
    reason = error.get('reason')
    if job_ref:
        message = f"Error processing job '{job_ref}': {error.get('message')}"
    else:
        message = error.get('message', '')
    new_errors = [err for err in error_ls if err != error]
    if new_errors:
        message += '\nFailure details:\n'
    wrap_error_message = True
    new_error_messages = [': '.join(filter(None, [err.get('location'), err.get('message')])) for err in new_errors]
    if wrap_error_message:
        message += '\n'.join((textwrap.fill(msg, initial_indent=' - ', subsequent_indent='   ') for msg in new_error_messages))
    else:
        error_message = '\n'.join(new_error_messages)
        if error_message:
            message += '- ' + error_message
    if session_id:
        message += '\nIn session: %s' % session_id
    message = bq_logging.EncodeForPrinting(message)
    if not reason or not message:
        return BigqueryInterfaceError('Error reported by server with missing error fields. Server returned: %s' % (str(server_error),))
    if reason == 'notFound':
        return BigqueryNotFoundError(message, error, error_ls, job_ref=job_ref)
    if reason == 'duplicate':
        return BigqueryDuplicateError(message, error, error_ls, job_ref=job_ref)
    if reason == 'accessDenied':
        return BigqueryAccessDeniedError(message, error, error_ls, job_ref=job_ref)
    if reason == 'invalidQuery':
        return BigqueryInvalidQueryError(message, error, error_ls, job_ref=job_ref)
    if reason == 'termsOfServiceNotAccepted':
        return BigqueryTermsOfServiceError(message, error, error_ls, job_ref=job_ref)
    if reason == 'backendError':
        return BigqueryBackendError(message, error, error_ls, job_ref=job_ref)
    return BigqueryServiceError(message, error, error_ls, job_ref=job_ref)