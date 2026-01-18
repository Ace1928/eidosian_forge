import re
from os import path
from nltk.corpus.reader import CorpusReader
from nltk.data import ZipFilePathPointer
from nltk.probability import FreqDist
def _load_lang_ngrams(self, lang):
    """Load single n-gram language file given the ISO 639-3 language code
        and return its FreqDist"""
    if lang not in self.langs():
        raise RuntimeError('Unsupported language.')
    crubadan_code = self.iso_to_crubadan(lang)
    ngram_file = path.join(self.root, crubadan_code + '-3grams.txt')
    if not path.isfile(ngram_file):
        raise RuntimeError('No N-gram file found for requested language.')
    counts = FreqDist()
    with open(ngram_file, encoding='utf-8') as f:
        for line in f:
            data = line.split(' ')
            ngram = data[1].strip('\n')
            freq = int(data[0])
            counts[ngram] = freq
    return counts