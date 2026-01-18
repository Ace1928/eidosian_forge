import textwrap
from typing import Dict, List, Optional
import bq_flags
from utils import bq_logging
class BigqueryDuplicateError(BigqueryServiceError):
    """The requested resource or identifier already exists."""