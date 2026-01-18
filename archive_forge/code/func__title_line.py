import Bio.GenBank
def _title_line(self):
    """Output for TITLE information (PRIVATE)."""
    output = ''
    if self.title:
        output += Record.INTERNAL_FORMAT % 'TITLE'
        output += _wrapped_genbank(self.title, Record.GB_BASE_INDENT)
    return output