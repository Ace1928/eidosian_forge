import Bio.GenBank
def _remark_line(self):
    """Output for REMARK information (PRIVATE)."""
    output = ''
    if self.remark:
        output += Record.INTERNAL_FORMAT % 'REMARK'
        output += _wrapped_genbank(self.remark, Record.GB_BASE_INDENT)
    return output