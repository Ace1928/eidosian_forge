import struct
from ..common.utils import struct_parse
from .sections import Section
class ELFHashTable(object):
    """ Representation of an ELF hash table to find symbols in the
        symbol table - useful for super-stripped binaries without section
        headers where only the start of the symbol table is known from the
        dynamic segment. The layout and contents are nicely described at
        https://flapenguin.me/2017/04/24/elf-lookup-dt-hash/.

        The symboltable argument needs to implement a get_symbol() method -
        in a regular ELF file, this will be the linked symbol table section
        as indicated by the sh_link attribute. For super-stripped binaries,
        one should use the DynamicSegment object as the symboltable as it
        supports symbol lookup without access to a symbol table section.
    """

    def __init__(self, elffile, start_offset, symboltable):
        self.elffile = elffile
        self._symboltable = symboltable
        self.params = struct_parse(self.elffile.structs.Elf_Hash, self.elffile.stream, start_offset)

    def get_number_of_symbols(self):
        """ Get the number of symbols from the hash table parameters.
        """
        return self.params['nchains']

    def get_symbol(self, name):
        """ Look up a symbol from this hash table with the given name.
        """
        if self.params['nbuckets'] == 0:
            return None
        hval = self.elf_hash(name) % self.params['nbuckets']
        symndx = self.params['buckets'][hval]
        while symndx != 0:
            sym = self._symboltable.get_symbol(symndx)
            if sym.name == name:
                return sym
            symndx = self.params['chains'][symndx]
        return None

    @staticmethod
    def elf_hash(name):
        """ Compute the hash value for a given symbol name.
        """
        if not isinstance(name, bytes):
            name = name.encode('utf-8')
        h = 0
        x = 0
        for c in bytearray(name):
            h = (h << 4) + c
            x = h & 4026531840
            if x != 0:
                h ^= x >> 24
            h &= ~x
        return h