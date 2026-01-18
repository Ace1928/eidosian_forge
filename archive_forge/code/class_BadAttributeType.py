import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
class BadAttributeType(ArffException):
    """Error raised when some invalid type is provided into the attribute
    declaration."""
    message = 'Bad @ATTRIBUTE type, at line %d.'