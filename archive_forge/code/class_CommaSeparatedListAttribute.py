import dataclasses
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
class CommaSeparatedListAttribute(_ListArrayAttribute):
    """For values which are sent to the server as a Comma Separated Values
    (CSV) string.  We allow them to be specified as a list and we convert it
    into a CSV"""