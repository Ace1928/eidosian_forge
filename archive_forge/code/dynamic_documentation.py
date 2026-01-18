import itertools
from collections import defaultdict
from .hash import ELFHashTable, GNUHashTable
from .sections import Section, Symbol
from .enums import ENUM_D_TAG
from .segments import Segment
from .relocation import RelocationTable, RelrRelocationTable
from ..common.exceptions import ELFError
from ..common.utils import elf_assert, struct_parse, parse_cstring_from_stream
 Yield all symbols in this dynamic segment. The symbols are usually
            the same as returned by SymbolTableSection.iter_symbols. However,
            in stripped binaries, SymbolTableSection might have been removed.
            This method reads from the mandatory dynamic tag DT_SYMTAB.
        