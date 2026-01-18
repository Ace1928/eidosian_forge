from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import sinica_parse
class SinicaTreebankCorpusReader(SyntaxCorpusReader):
    """
    Reader for the sinica treebank.
    """

    def _read_block(self, stream):
        sent = stream.readline()
        sent = IDENTIFIER.sub('', sent)
        sent = APPENDIX.sub('', sent)
        return [sent]

    def _parse(self, sent):
        return sinica_parse(sent)

    def _tag(self, sent, tagset=None):
        tagged_sent = [(w, t) for t, w in TAGWORD.findall(sent)]
        if tagset and tagset != self._tagset:
            tagged_sent = [(w, map_tag(self._tagset, tagset, t)) for w, t in tagged_sent]
        return tagged_sent

    def _word(self, sent):
        return WORD.findall(sent)