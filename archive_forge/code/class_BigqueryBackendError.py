import textwrap
from typing import Dict, List, Optional
import bq_flags
from utils import bq_logging
class BigqueryBackendError(BigqueryServiceError):
    """A backend error typically corresponding to retriable HTTP 5xx failures."""