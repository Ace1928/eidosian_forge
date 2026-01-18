from Bio.KEGG import _default_wrap, _wrap_kegg, _write_kegg
def _definition(self):
    return _write_kegg('DEFINITION', [self.definition])