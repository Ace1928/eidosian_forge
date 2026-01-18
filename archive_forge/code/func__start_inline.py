import logging
def _start_inline(self, markup):
    try:
        last_write = self.doc.peek_write()
    except IndexError:
        pass
    else:
        if last_write in ('*', '**') and markup in ('*', '**'):
            self.doc.write(' ')
    self.doc.write(markup)