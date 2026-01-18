import Bio.GenBank
def _db_source_line(self):
    """Output for DBSOURCE line (PRIVATE)."""
    if self.db_source:
        output = Record.BASE_FORMAT % 'DBSOURCE'
        output += f'{self.db_source}\n'
    else:
        output = ''
    return output