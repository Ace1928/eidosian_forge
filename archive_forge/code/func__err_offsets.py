import unicodedata
def _err_offsets(self):
    lineno = 1
    colno = 1
    for i in range(self.errpos):
        if self.msg[i] == '\n':
            lineno += 1
            colno = 1
        else:
            colno += 1
    return (lineno, colno)