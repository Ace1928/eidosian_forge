from Bio.KEGG import _default_wrap, _struct_wrap, _wrap_kegg, _write_kegg
def _substrate(self):
    return _write_kegg('SUBSTRATE', [_wrap_kegg(line, wrap_rule=name_wrap) for line in self.substrate])