from ..construct import CString
from ..common.utils import struct_parse, elf_assert
from .sections import Section, Symbol
class GNUVersionSection(Section):
    """ Common ancestor class for ELF SUNW|GNU Version Needed/Dependency
        sections class which contains shareable code
    """

    def __init__(self, header, name, elffile, stringtable, field_prefix, version_struct, version_auxiliaries_struct):
        super(GNUVersionSection, self).__init__(header, name, elffile)
        self.stringtable = stringtable
        self.field_prefix = field_prefix
        self.version_struct = version_struct
        self.version_auxiliaries_struct = version_auxiliaries_struct

    def num_versions(self):
        """ Number of version entries in the section
        """
        return self['sh_info']

    def _field_name(self, name, auxiliary=False):
        """ Return the real field's name of version or a version auxiliary
            entry
        """
        middle = 'a_' if auxiliary else '_'
        return self.field_prefix + middle + name

    def _iter_version_auxiliaries(self, entry_offset, count):
        """ Yield all auxiliary entries of a version entry
        """
        name_field = self._field_name('name', auxiliary=True)
        next_field = self._field_name('next', auxiliary=True)
        for _ in range(count):
            entry = struct_parse(self.version_auxiliaries_struct, self.stream, stream_pos=entry_offset)
            name = self.stringtable.get_string(entry[name_field])
            version_aux = VersionAuxiliary(entry, name)
            yield version_aux
            entry_offset += entry[next_field]

    def iter_versions(self):
        """ Yield all the version entries in the section
            Each time it returns the main version structure
            and an iterator to walk through its auxiliaries entries
        """
        aux_field = self._field_name('aux')
        count_field = self._field_name('cnt')
        next_field = self._field_name('next')
        entry_offset = self['sh_offset']
        for _ in range(self.num_versions()):
            entry = struct_parse(self.version_struct, self.stream, stream_pos=entry_offset)
            elf_assert(entry[count_field] > 0, 'Expected number of version auxiliary entries (%s) to be > 0for the following version entry: %s' % (count_field, str(entry)))
            version = Version(entry)
            aux_entries_offset = entry_offset + entry[aux_field]
            version_auxiliaries_iter = self._iter_version_auxiliaries(aux_entries_offset, entry[count_field])
            yield (version, version_auxiliaries_iter)
            entry_offset += entry[next_field]