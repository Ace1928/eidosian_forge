import textwrap
from typing import Dict, List, Optional
import bq_flags
from utils import bq_logging
class BigqueryTermsOfServiceError(BigqueryAccessDeniedError):
    """User has not ACK'd ToS."""