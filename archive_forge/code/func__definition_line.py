import Bio.GenBank
def _definition_line(self):
    """Provide output for the DEFINITION line (PRIVATE)."""
    output = Record.BASE_FORMAT % 'DEFINITION'
    output += _wrapped_genbank(self.definition + '.', Record.GB_BASE_INDENT)
    return output