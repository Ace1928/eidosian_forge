import re
from collections import defaultdict
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import ElementTree, XMLCorpusReader
from nltk.util import LazyConcatenation, LazyMap, flatten
def _getMLU(self, fileid, speaker):
    sents = self._get_words(fileid, speaker=speaker, sent=True, stem=True, relation=False, pos=True, strip_space=True, replace=True)
    results = []
    lastSent = []
    numFillers = 0
    sentDiscount = 0
    for sent in sents:
        posList = [pos for word, pos in sent]
        if any((pos == 'unk' for pos in posList)):
            continue
        elif sent == []:
            continue
        elif sent == lastSent:
            continue
        else:
            results.append([word for word, pos in sent])
            if len({'co', None}.intersection(posList)) > 0:
                numFillers += posList.count('co')
                numFillers += posList.count(None)
                sentDiscount += 1
        lastSent = sent
    try:
        thisWordList = flatten(results)
        numWords = len(flatten([word.split('-') for word in thisWordList])) - numFillers
        numSents = len(results) - sentDiscount
        mlu = numWords / numSents
    except ZeroDivisionError:
        mlu = 0
    return mlu