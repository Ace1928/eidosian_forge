import textwrap
from typing import Dict, List, Optional
import bq_flags
from utils import bq_logging
class BigqueryAccessDeniedError(BigqueryServiceError):
    """The user does not have access to the requested resource."""