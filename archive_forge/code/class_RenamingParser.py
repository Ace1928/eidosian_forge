from formencode.rewritingparser import RewritingParser
class RenamingParser(RewritingParser):

    def __init__(self, rename_func):
        RewritingParser.__init__(self)
        self.rename_func = rename_func

    def close(self):
        self.handle_misc(None)
        RewritingParser.close(self)
        self._text = self._get_text()

    def text(self):
        try:
            return self._text
        except AttributeError:
            raise Exception('You must .close() a parser instance before getting the text from it')

    def handle_starttag(self, tag, attrs, startend=False):
        self.write_pos()
        if tag in ('input', 'textarea', 'select'):
            self.handle_field(tag, attrs, startend)
        else:
            return

    def handle_startendtag(self, tag, attrs):
        return self.handle_starttag(tag, attrs, True)

    def handle_field(self, tag, attrs, startend):
        name = self.get_attr(attrs, 'name', '')
        new_name = self.rename_func(name)
        if name is None:
            self.del_attr(attrs, 'name')
        else:
            self.set_attr(attrs, 'name', new_name)
        self.write_tag(tag, attrs)
        self.skip_next = True