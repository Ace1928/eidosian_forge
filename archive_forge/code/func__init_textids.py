from nltk.corpus.reader.api import *
from nltk.corpus.reader.xmldocs import XMLCorpusReader
def _init_textids(self):
    self._f2t = defaultdict(list)
    self._t2f = defaultdict(list)
    if self._textids is not None:
        with open(self._textids) as fp:
            for line in fp:
                line = line.strip()
                file_id, text_ids = line.split(' ', 1)
                if file_id not in self.fileids():
                    raise ValueError('In text_id mapping file %s: %s not found' % (self._textids, file_id))
                for text_id in text_ids.split(self._delimiter):
                    self._add_textids(file_id, text_id)