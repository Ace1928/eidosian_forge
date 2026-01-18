import textwrap
from typing import Dict, List, Optional
import bq_flags
from utils import bq_logging
class BigquerySchemaError(BigqueryClientError):
    """Error in locating or parsing the schema."""