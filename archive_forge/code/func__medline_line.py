import Bio.GenBank
def _medline_line(self):
    """Output for MEDLINE information (PRIVATE)."""
    output = ''
    if self.medline_id:
        output += Record.INTERNAL_FORMAT % 'MEDLINE'
        output += self.medline_id + '\n'
    return output