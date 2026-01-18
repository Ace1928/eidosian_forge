import Bio.GenBank
def _source_line(self):
    """Output for SOURCE line on where the sample came from (PRIVATE)."""
    output = Record.BASE_FORMAT % 'SOURCE'
    output += _wrapped_genbank(self.source, Record.GB_BASE_INDENT)
    return output