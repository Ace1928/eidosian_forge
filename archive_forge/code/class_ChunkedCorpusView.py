import codecs
import os.path
import nltk
from nltk.chunk import tagstr2tree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from nltk.corpus.reader.util import *
from nltk.tokenize import *
from nltk.tree import Tree
class ChunkedCorpusView(StreamBackedCorpusView):

    def __init__(self, fileid, encoding, tagged, group_by_sent, group_by_para, chunked, str2chunktree, sent_tokenizer, para_block_reader, source_tagset=None, target_tagset=None):
        StreamBackedCorpusView.__init__(self, fileid, encoding=encoding)
        self._tagged = tagged
        self._group_by_sent = group_by_sent
        self._group_by_para = group_by_para
        self._chunked = chunked
        self._str2chunktree = str2chunktree
        self._sent_tokenizer = sent_tokenizer
        self._para_block_reader = para_block_reader
        self._source_tagset = source_tagset
        self._target_tagset = target_tagset

    def read_block(self, stream):
        block = []
        for para_str in self._para_block_reader(stream):
            para = []
            for sent_str in self._sent_tokenizer.tokenize(para_str):
                sent = self._str2chunktree(sent_str, source_tagset=self._source_tagset, target_tagset=self._target_tagset)
                if not self._tagged:
                    sent = self._untag(sent)
                if not self._chunked:
                    sent = sent.leaves()
                if self._group_by_sent:
                    para.append(sent)
                else:
                    para.extend(sent)
            if self._group_by_para:
                block.append(para)
            else:
                block.extend(para)
        return block

    def _untag(self, tree):
        for i, child in enumerate(tree):
            if isinstance(child, Tree):
                self._untag(child)
            elif isinstance(child, tuple):
                tree[i] = child[0]
            else:
                raise ValueError('expected child to be Tree or tuple')
        return tree