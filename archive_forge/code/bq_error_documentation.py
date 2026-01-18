import textwrap
from typing import Dict, List, Optional
import bq_flags
from utils import bq_logging
Initializes a BigqueryServiceError.

    Args:
      message: A user-facing error message.
      error: The error dictionary, code may inspect the 'reason' key.
      error_list: A list of additional entries, for example a load job may
        contain multiple errors here for each error encountered during
        processing.
      job_ref: Optional JobReference string, if this error was encountered while
        processing a job.
    