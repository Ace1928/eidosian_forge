class CDefError(Exception):
    __module__ = 'cffi'

    def __str__(self):
        try:
            current_decl = self.args[1]
            filename = current_decl.coord.file
            linenum = current_decl.coord.line
            prefix = '%s:%d: ' % (filename, linenum)
        except (AttributeError, TypeError, IndexError):
            prefix = ''
        return '%s%s' % (prefix, self.args[0])