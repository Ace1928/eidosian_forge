import textwrap
from typing import Dict, List, Optional
import bq_flags
from utils import bq_logging
class BigqueryNotFoundError(BigqueryServiceError):
    """The requested resource or identifier was not found."""