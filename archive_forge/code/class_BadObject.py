import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
class BadObject(ArffException):
    """Error raised when the object representing the ARFF file has something
    wrong."""

    def __init__(self, msg='Invalid object.'):
        self.msg = msg

    def __str__(self):
        return '%s' % self.msg