import Bio.GenBank
def _dblink_line(self):
    output = ''
    if len(self.dblinks) > 0:
        output = Record.BASE_FORMAT % 'DBLINK'
        dblink_info = '\n'.join(self.dblinks)
        output += _wrapped_genbank(dblink_info, Record.GB_BASE_INDENT)
    return output