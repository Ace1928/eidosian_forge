import ast
import re
from abc import abstractmethod
from typing import List, Optional, Tuple
from nltk import jsontags
from nltk.classify import NaiveBayesClassifier
from nltk.probability import ConditionalFreqDist
from nltk.tag.api import FeaturesetTaggerI, TaggerI
@jsontags.register_tag
class AffixTagger(ContextTagger):
    """
    A tagger that chooses a token's tag based on a leading or trailing
    substring of its word string.  (It is important to note that these
    substrings are not necessarily "true" morphological affixes).  In
    particular, a fixed-length substring of the word is looked up in a
    table, and the corresponding tag is returned.  Affix taggers are
    typically constructed by training them on a tagged corpus.

    Construct a new affix tagger.

    :param affix_length: The length of the affixes that should be
        considered during training and tagging.  Use negative
        numbers for suffixes.
    :param min_stem_length: Any words whose length is less than
        min_stem_length+abs(affix_length) will be assigned a
        tag of None by this tagger.
    """
    json_tag = 'nltk.tag.sequential.AffixTagger'

    def __init__(self, train=None, model=None, affix_length=-3, min_stem_length=2, backoff=None, cutoff=0, verbose=False):
        self._check_params(train, model)
        super().__init__(model, backoff)
        self._affix_length = affix_length
        self._min_word_length = min_stem_length + abs(affix_length)
        if train:
            self._train(train, cutoff, verbose)

    def encode_json_obj(self):
        return (self._affix_length, self._min_word_length, self._context_to_tag, self.backoff)

    @classmethod
    def decode_json_obj(cls, obj):
        _affix_length, _min_word_length, _context_to_tag, backoff = obj
        return cls(affix_length=_affix_length, min_stem_length=_min_word_length - abs(_affix_length), model=_context_to_tag, backoff=backoff)

    def context(self, tokens, index, history):
        token = tokens[index]
        if len(token) < self._min_word_length:
            return None
        elif self._affix_length > 0:
            return token[:self._affix_length]
        else:
            return token[self._affix_length:]