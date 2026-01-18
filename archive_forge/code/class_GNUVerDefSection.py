from ..construct import CString
from ..common.utils import struct_parse, elf_assert
from .sections import Section, Symbol
class GNUVerDefSection(GNUVersionSection):
    """ ELF SUNW or GNU Version Definition table section.
        Has an associated StringTableSection that's passed in the constructor.
    """

    def __init__(self, header, name, elffile, stringtable):
        super(GNUVerDefSection, self).__init__(header, name, elffile, stringtable, 'vd', elffile.structs.Elf_Verdef, elffile.structs.Elf_Verdaux)

    def get_version(self, index):
        """ Get the version information located at index #n in the table
            Return boths the verdef structure and an iterator to retrieve
            both the version names and dependencies in the form of
            verdaux entries
        """
        for verdef, verdaux_iter in self.iter_versions():
            if verdef['vd_ndx'] == index:
                return (verdef, verdaux_iter)
        return None