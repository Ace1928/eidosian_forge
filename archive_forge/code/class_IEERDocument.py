import nltk
from nltk.corpus.reader.api import *
class IEERDocument:

    def __init__(self, text, docno=None, doctype=None, date_time=None, headline=''):
        self.text = text
        self.docno = docno
        self.doctype = doctype
        self.date_time = date_time
        self.headline = headline

    def __repr__(self):
        if self.headline:
            headline = ' '.join(self.headline.leaves())
        else:
            headline = ' '.join([w for w in self.text.leaves() if w[:1] != '<'][:12]) + '...'
        if self.docno is not None:
            return f'<IEERDocument {self.docno}: {headline!r}>'
        else:
            return '<IEERDocument: %r>' % headline