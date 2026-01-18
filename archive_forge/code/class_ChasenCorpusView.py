import sys
from nltk.corpus.reader import util
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
class ChasenCorpusView(StreamBackedCorpusView):
    """
    A specialized corpus view for ChasenReader. Similar to ``TaggedCorpusView``,
    but this'll use fixed sets of word and sentence tokenizer.
    """

    def __init__(self, corpus_file, encoding, tagged, group_by_sent, group_by_para, sent_splitter=None):
        self._tagged = tagged
        self._group_by_sent = group_by_sent
        self._group_by_para = group_by_para
        self._sent_splitter = sent_splitter
        StreamBackedCorpusView.__init__(self, corpus_file, encoding=encoding)

    def read_block(self, stream):
        """Reads one paragraph at a time."""
        block = []
        for para_str in read_regexp_block(stream, '.', '^EOS\\n'):
            para = []
            sent = []
            for line in para_str.splitlines():
                _eos = line.strip() == 'EOS'
                _cells = line.split('\t')
                w = (_cells[0], '\t'.join(_cells[1:]))
                if not _eos:
                    sent.append(w)
                if _eos or (self._sent_splitter and self._sent_splitter(w)):
                    if not self._tagged:
                        sent = [w for w, t in sent]
                    if self._group_by_sent:
                        para.append(sent)
                    else:
                        para.extend(sent)
                    sent = []
            if len(sent) > 0:
                if not self._tagged:
                    sent = [w for w, t in sent]
                if self._group_by_sent:
                    para.append(sent)
                else:
                    para.extend(sent)
            if self._group_by_para:
                block.append(para)
            else:
                block.extend(para)
        return block