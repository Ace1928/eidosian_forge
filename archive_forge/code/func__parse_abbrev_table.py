from ..common.utils import struct_parse, dwarf_assert
def _parse_abbrev_table(self):
    """ Parse the abbrev table from the stream
        """
    map = {}
    self.stream.seek(self.offset)
    while True:
        decl_code = struct_parse(struct=self.structs.Dwarf_uleb128(''), stream=self.stream)
        if decl_code == 0:
            break
        declaration = struct_parse(struct=self.structs.Dwarf_abbrev_declaration, stream=self.stream)
        map[decl_code] = AbbrevDecl(decl_code, declaration)
    return map