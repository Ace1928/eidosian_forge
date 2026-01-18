from nltk.corpus.reader.api import *
from nltk.corpus.reader.xmldocs import XMLCorpusReader
class TEICorpusView(StreamBackedCorpusView):

    def __init__(self, corpus_file, tagged, group_by_sent, group_by_para, tagset=None, head_len=0, textids=None):
        self._tagged = tagged
        self._textids = textids
        self._group_by_sent = group_by_sent
        self._group_by_para = group_by_para
        StreamBackedCorpusView.__init__(self, corpus_file, startpos=head_len)
    _pagesize = 4096

    def read_block(self, stream):
        block = stream.readlines(self._pagesize)
        block = concat(block)
        while block.count('<text id') > block.count('</text>') or block.count('<text id') == 0:
            tmp = stream.readline()
            if len(tmp) <= 0:
                break
            block += tmp
        block = block.replace('\n', '')
        textids = TEXTID.findall(block)
        if self._textids:
            for tid in textids:
                if tid not in self._textids:
                    beg = block.find(tid) - 1
                    end = block[beg:].find('</text>') + len('</text>')
                    block = block[:beg] + block[beg + end:]
        output = []
        for para_str in PARA.findall(block):
            para = []
            for sent_str in SENT.findall(para_str):
                if not self._tagged:
                    sent = WORD.findall(sent_str)
                else:
                    sent = list(map(self._parse_tag, TAGGEDWORD.findall(sent_str)))
                if self._group_by_sent:
                    para.append(sent)
                else:
                    para.extend(sent)
            if self._group_by_para:
                output.append(para)
            else:
                output.extend(para)
        return output

    def _parse_tag(self, tag_word_tuple):
        tag, word = tag_word_tuple
        if tag.startswith('w'):
            tag = ANA.search(tag).group(1)
        else:
            tag = TYPE.search(tag).group(1)
        return (word, tag)