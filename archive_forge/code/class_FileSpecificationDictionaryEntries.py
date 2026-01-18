from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
class FileSpecificationDictionaryEntries:
    """TABLE 3.41 Entries in a file specification dictionary."""
    Type = '/Type'
    FS = '/FS'
    F = '/F'
    UF = '/UF'
    DOS = '/DOS'
    Mac = '/Mac'
    Unix = '/Unix'
    ID = '/ID'
    V = '/V'
    EF = '/EF'
    RF = '/RF'
    DESC = '/Desc'
    Cl = '/Cl'