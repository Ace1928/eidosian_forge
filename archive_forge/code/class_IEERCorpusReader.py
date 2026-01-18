import nltk
from nltk.corpus.reader.api import *
class IEERCorpusReader(CorpusReader):
    """ """

    def docs(self, fileids=None):
        return concat([StreamBackedCorpusView(fileid, self._read_block, encoding=enc) for fileid, enc in self.abspaths(fileids, True)])

    def parsed_docs(self, fileids=None):
        return concat([StreamBackedCorpusView(fileid, self._read_parsed_block, encoding=enc) for fileid, enc in self.abspaths(fileids, True)])

    def _read_parsed_block(self, stream):
        return [self._parse(doc) for doc in self._read_block(stream) if self._parse(doc).docno is not None]

    def _parse(self, doc):
        val = nltk.chunk.ieerstr2tree(doc, root_label='DOCUMENT')
        if isinstance(val, dict):
            return IEERDocument(**val)
        else:
            return IEERDocument(val)

    def _read_block(self, stream):
        out = []
        while True:
            line = stream.readline()
            if not line:
                break
            if line.strip() == '<DOC>':
                break
        out.append(line)
        while True:
            line = stream.readline()
            if not line:
                break
            out.append(line)
            if line.strip() == '</DOC>':
                break
        return ['\n'.join(out)]