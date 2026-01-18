import textwrap
from typing import Dict, List, Optional
import bq_flags
from utils import bq_logging
class BigqueryCommunicationError(BigqueryError):
    """Error communicating with the server."""