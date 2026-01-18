from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
class PPAttachment:

    def __init__(self, sent, verb, noun1, prep, noun2, attachment):
        self.sent = sent
        self.verb = verb
        self.noun1 = noun1
        self.prep = prep
        self.noun2 = noun2
        self.attachment = attachment

    def __repr__(self):
        return 'PPAttachment(sent=%r, verb=%r, noun1=%r, prep=%r, noun2=%r, attachment=%r)' % (self.sent, self.verb, self.noun1, self.prep, self.noun2, self.attachment)