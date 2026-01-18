from ..construct import CString
from ..common.utils import struct_parse, elf_assert
from .sections import Section, Symbol
class VersionAuxiliary(object):
    """ Version Auxiliary object - representing an auxiliary entry of a version
        definition or dependency entry

        Similarly to Section objects, allows dictionary-like access to the
        verdaux/vernaux entry
    """

    def __init__(self, entry, name):
        self.entry = entry
        self.name = name

    def __getitem__(self, name):
        """ Implement dict-like access to entries
        """
        return self.entry[name]