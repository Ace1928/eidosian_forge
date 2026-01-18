import Bio.GenBank
def _authors_line(self):
    """Output for AUTHORS information (PRIVATE)."""
    output = ''
    if self.authors:
        output += Record.INTERNAL_FORMAT % 'AUTHORS'
        output += _wrapped_genbank(self.authors, Record.GB_BASE_INDENT)
    return output