import Bio.GenBank
def _accession_line(self):
    """Output for the ACCESSION line (PRIVATE)."""
    if self.accession:
        output = Record.BASE_FORMAT % 'ACCESSION'
        acc_info = ''
        for accession in self.accession:
            acc_info += f'{accession} '
        acc_info = acc_info.rstrip()
        output += _wrapped_genbank(acc_info, Record.GB_BASE_INDENT)
    else:
        output = ''
    return output