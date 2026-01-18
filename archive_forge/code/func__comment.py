from Bio.KEGG import _default_wrap, _struct_wrap, _wrap_kegg, _write_kegg
def _comment(self):
    return _write_kegg('COMMENT', [_wrap_kegg(line, wrap_rule=id_wrap(0)) for line in self.comment])