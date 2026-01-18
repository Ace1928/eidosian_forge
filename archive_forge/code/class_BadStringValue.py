import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
class BadStringValue(ArffException):
    """Error raise when a string contains space but is not quoted."""
    message = 'Invalid string value at line %d.'