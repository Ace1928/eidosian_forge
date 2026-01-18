import Bio.GenBank
def _contig_line(self):
    """Output for CONTIG location information from RefSeq (PRIVATE)."""
    output = ''
    if self.contig:
        output += Record.BASE_FORMAT % 'CONTIG'
        output += _wrapped_genbank(self.contig, Record.GB_BASE_INDENT, split_char=',')
    return output