import Bio.GenBank
def _segment_line(self):
    """Output for the SEGMENT line (PRIVATE)."""
    output = ''
    if self.segment:
        output += Record.BASE_FORMAT % 'SEGMENT'
        output += _wrapped_genbank(self.segment, Record.GB_BASE_INDENT)
    return output