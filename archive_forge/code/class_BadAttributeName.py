import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
class BadAttributeName(ArffException):
    """Error raised when an attribute name is provided twice the attribute
    declaration."""

    def __init__(self, value, value2):
        super().__init__()
        self.message = 'Bad @ATTRIBUTE name %s at line' % value + ' %d, this name is already in use in line' + ' %d.' % value2