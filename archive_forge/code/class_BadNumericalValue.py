import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
class BadNumericalValue(ArffException):
    """Error raised when and invalid numerical value is used in some data
    instance."""
    message = 'Invalid numerical value, at line %d.'