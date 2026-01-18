from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
class SymbolTableSection(Section):
    """ ELF symbol table section. Has an associated StringTableSection that's
        passed in the constructor.
    """

    def __init__(self, header, name, elffile, stringtable):
        super(SymbolTableSection, self).__init__(header, name, elffile)
        self.stringtable = stringtable
        elf_assert(self['sh_entsize'] > 0, 'Expected entry size of section %r to be > 0' % name)
        elf_assert(self['sh_size'] % self['sh_entsize'] == 0, 'Expected section size to be a multiple of entry size in section %r' % name)
        self._symbol_name_map = None

    def num_symbols(self):
        """ Number of symbols in the table
        """
        return self['sh_size'] // self['sh_entsize']

    def get_symbol(self, n):
        """ Get the symbol at index #n from the table (Symbol object)
        """
        entry_offset = self['sh_offset'] + n * self['sh_entsize']
        entry = struct_parse(self.structs.Elf_Sym, self.stream, stream_pos=entry_offset)
        name = self.stringtable.get_string(entry['st_name'])
        return Symbol(entry, name)

    def get_symbol_by_name(self, name):
        """ Get a symbol(s) by name. Return None if no symbol by the given name
            exists.
        """
        if self._symbol_name_map is None:
            self._symbol_name_map = defaultdict(list)
            for i, sym in enumerate(self.iter_symbols()):
                self._symbol_name_map[sym.name].append(i)
        symnums = self._symbol_name_map.get(name)
        return [self.get_symbol(i) for i in symnums] if symnums else None

    def iter_symbols(self):
        """ Yield all the symbols in the table
        """
        for i in range(self.num_symbols()):
            yield self.get_symbol(i)