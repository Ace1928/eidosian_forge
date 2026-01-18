import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
class BadRelationFormat(ArffException):
    """Error raised when the relation declaration is in an invalid format."""
    message = 'Bad @RELATION format, at line %d.'