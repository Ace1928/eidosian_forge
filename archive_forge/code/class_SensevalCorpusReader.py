import re
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import *
class SensevalCorpusReader(CorpusReader):

    def instances(self, fileids=None):
        return concat([SensevalCorpusView(fileid, enc) for fileid, enc in self.abspaths(fileids, True)])

    def _entry(self, tree):
        elts = []
        for lexelt in tree.findall('lexelt'):
            for inst in lexelt.findall('instance'):
                sense = inst[0].attrib['senseid']
                context = [(w.text, w.attrib['pos']) for w in inst[1]]
                elts.append((sense, context))
        return elts