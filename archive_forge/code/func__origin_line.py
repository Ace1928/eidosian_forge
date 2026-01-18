import Bio.GenBank
def _origin_line(self):
    """Output for the ORIGIN line (PRIVATE)."""
    output = ''
    if self.sequence:
        output += Record.BASE_FORMAT % 'ORIGIN'
        if self.origin:
            output += _wrapped_genbank(self.origin, Record.GB_BASE_INDENT)
        else:
            output += '\n'
    return output