import _markupbase
import re
class TestSGMLParser(SGMLParser):

    def __init__(self, verbose=0):
        self.testdata = ''
        SGMLParser.__init__(self, verbose)

    def handle_data(self, data):
        self.testdata = self.testdata + data
        if len(repr(self.testdata)) >= 70:
            self.flush()

    def flush(self):
        data = self.testdata
        if data:
            self.testdata = ''
            print('data:', repr(data))

    def handle_comment(self, data):
        self.flush()
        r = repr(data)
        if len(r) > 68:
            r = r[:32] + '...' + r[-32:]
        print('comment:', r)

    def unknown_starttag(self, tag, attrs):
        self.flush()
        if not attrs:
            print('start tag: <' + tag + '>')
        else:
            print('start tag: <' + tag, end=' ')
            for name, value in attrs:
                print(name + '=' + '"' + value + '"', end=' ')
            print('>')

    def unknown_endtag(self, tag):
        self.flush()
        print('end tag: </' + tag + '>')

    def unknown_entityref(self, ref):
        self.flush()
        print('*** unknown entity ref: &' + ref + ';')

    def unknown_charref(self, ref):
        self.flush()
        print('*** unknown char ref: &#' + ref + ';')

    def unknown_decl(self, data):
        self.flush()
        print('*** unknown decl: [' + data + ']')

    def close(self):
        SGMLParser.close(self)
        self.flush()