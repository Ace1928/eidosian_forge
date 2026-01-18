import logging
def _end_inline(self, markup):
    last_write = self.doc.pop_write()
    if last_write == markup:
        return
    self.doc.push_write(last_write)
    self.doc.write(markup)