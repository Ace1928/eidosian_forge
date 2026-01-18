import textwrap
from typing import Dict, List, Optional
import bq_flags
from utils import bq_logging
class BigqueryServiceError(BigqueryError):
    """Base class of Bigquery-specific error responses.

  The BigQuery server received request and returned an error.
  """

    def __init__(self, message: str, error: Dict[str, str], error_list: List[Dict[str, str]], job_ref: Optional[str]=None, *args, **kwds):
        """Initializes a BigqueryServiceError.

    Args:
      message: A user-facing error message.
      error: The error dictionary, code may inspect the 'reason' key.
      error_list: A list of additional entries, for example a load job may
        contain multiple errors here for each error encountered during
        processing.
      job_ref: Optional JobReference string, if this error was encountered while
        processing a job.
    """
        super().__init__(message, *args, **kwds)
        self.error = error
        self.error_list = error_list
        self.job_ref = job_ref

    def __repr__(self):
        return '%s: error=%s, error_list=%s, job_ref=%s' % (self.__class__.__name__, self.error, self.error_list, self.job_ref)