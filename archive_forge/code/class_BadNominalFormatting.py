import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
class BadNominalFormatting(ArffException):
    """Error raised when a nominal value with space is not properly quoted."""

    def __init__(self, value):
        super().__init__()
        self.message = 'Nominal data value "%s" not properly quoted in line ' % value + '%d.'