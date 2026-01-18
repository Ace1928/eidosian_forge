import sys
import re
import operator
import typing
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
class SelectorSyntaxError(SelectorError, SyntaxError):
    """Parsing a selector that does not match the grammar."""