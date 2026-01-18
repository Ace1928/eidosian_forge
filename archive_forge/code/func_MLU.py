import re
from collections import defaultdict
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import ElementTree, XMLCorpusReader
from nltk.util import LazyConcatenation, LazyMap, flatten
def MLU(self, fileids=None, speaker='CHI'):
    """
        :return: the given file(s) as a floating number
        :rtype: list(float)
        """
    if not self._lazy:
        return [self._getMLU(fileid, speaker=speaker) for fileid in self.abspaths(fileids)]
    get_MLU = lambda fileid: self._getMLU(fileid, speaker=speaker)
    return LazyMap(get_MLU, self.abspaths(fileids))