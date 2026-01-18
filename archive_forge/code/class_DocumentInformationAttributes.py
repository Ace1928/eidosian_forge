from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
class DocumentInformationAttributes:
    """TABLE 10.2 Entries in the document information dictionary."""
    TITLE = '/Title'
    AUTHOR = '/Author'
    SUBJECT = '/Subject'
    KEYWORDS = '/Keywords'
    CREATOR = '/Creator'
    PRODUCER = '/Producer'
    CREATION_DATE = '/CreationDate'
    MOD_DATE = '/ModDate'
    TRAPPED = '/Trapped'