import Bio.GenBank
def _keywords_line(self):
    """Output for the KEYWORDS line (PRIVATE)."""
    output = ''
    if self.keywords:
        output += Record.BASE_FORMAT % 'KEYWORDS'
        keyword_info = ''
        for keyword in self.keywords:
            keyword_info += f'{keyword}; '
        keyword_info = keyword_info[:-2]
        keyword_info += '.'
        output += _wrapped_genbank(keyword_info, Record.GB_BASE_INDENT)
    return output