import Bio.GenBank
def _locus_line(self):
    """Provide the output string for the LOCUS line (PRIVATE)."""
    output = 'LOCUS'
    output += ' ' * 7
    output += '%-9s' % self.locus
    output += ' '
    output += '%7s' % self.size
    if 'PROTEIN' in self.residue_type:
        output += ' aa'
    else:
        output += ' bp '
    if 'circular' in self.residue_type:
        output += '%17s' % self.residue_type
    elif '-' in self.residue_type:
        output += '%7s' % self.residue_type
        output += ' ' * 10
    else:
        output += ' ' * 3
        output += '%-4s' % self.residue_type
        output += ' ' * 10
    output += ' ' * 2
    output += '%3s' % self.data_file_division
    output += ' ' * 7
    output += '%11s' % self.date
    output += '\n'
    return output