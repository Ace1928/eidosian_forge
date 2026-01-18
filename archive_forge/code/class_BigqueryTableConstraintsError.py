import textwrap
from typing import Dict, List, Optional
import bq_flags
from utils import bq_logging
class BigqueryTableConstraintsError(BigqueryClientError):
    """Error in locating or parsing the table constraints."""