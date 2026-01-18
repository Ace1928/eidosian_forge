import Bio.GenBank
def _consrtm_line(self):
    """Output for CONSRTM information (PRIVATE)."""
    output = ''
    if self.consrtm:
        output += Record.INTERNAL_FORMAT % 'CONSRTM'
        output += _wrapped_genbank(self.consrtm, Record.GB_BASE_INDENT)
    return output