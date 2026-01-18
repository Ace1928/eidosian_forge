from typing import Iterator, List, Optional, Tuple, cast
from ._protocols import PdfCommonDocProtocol
from ._utils import logger_warning
from .generic import ArrayObject, DictionaryObject, NullObject, NumberObject

    Return the (key, value) pair of the entry after the given one.

    See 7.9.7 "Number Trees".

    Args:
        key: number key of the entry
        nums: Nums array
    