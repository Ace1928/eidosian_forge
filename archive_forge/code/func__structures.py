from Bio.KEGG import _default_wrap, _struct_wrap, _wrap_kegg, _write_kegg
def _structures(self):
    s = []
    for entry in self.structures:
        s.append(entry[0] + ': ' + '  '.join(entry[1]) + '  ')
    return _write_kegg('STRUCTURES', [_wrap_kegg(line, wrap_rule=struct_wrap(5)) for line in s])