import ast
import re
from abc import abstractmethod
from typing import List, Optional, Tuple
from nltk import jsontags
from nltk.classify import NaiveBayesClassifier
from nltk.probability import ConditionalFreqDist
from nltk.tag.api import FeaturesetTaggerI, TaggerI
@jsontags.register_tag
class NgramTagger(ContextTagger):
    """
    A tagger that chooses a token's tag based on its word string and
    on the preceding n word's tags.  In particular, a tuple
    (tags[i-n:i-1], words[i]) is looked up in a table, and the
    corresponding tag is returned.  N-gram taggers are typically
    trained on a tagged corpus.

    Train a new NgramTagger using the given training data or
    the supplied model.  In particular, construct a new tagger
    whose table maps from each context (tag[i-n:i-1], word[i])
    to the most frequent tag for that context.  But exclude any
    contexts that are already tagged perfectly by the backoff
    tagger.

    :param train: A tagged corpus consisting of a list of tagged
        sentences, where each sentence is a list of (word, tag) tuples.
    :param backoff: A backoff tagger, to be used by the new
        tagger if it encounters an unknown context.
    :param cutoff: If the most likely tag for a context occurs
        fewer than *cutoff* times, then exclude it from the
        context-to-tag table for the new tagger.
    """
    json_tag = 'nltk.tag.sequential.NgramTagger'

    def __init__(self, n, train=None, model=None, backoff=None, cutoff=0, verbose=False):
        self._n = n
        self._check_params(train, model)
        super().__init__(model, backoff)
        if train:
            self._train(train, cutoff, verbose)

    def encode_json_obj(self):
        _context_to_tag = {repr(k): v for k, v in self._context_to_tag.items()}
        if 'NgramTagger' in self.__class__.__name__:
            return (self._n, _context_to_tag, self.backoff)
        else:
            return (_context_to_tag, self.backoff)

    @classmethod
    def decode_json_obj(cls, obj):
        try:
            _n, _context_to_tag, backoff = obj
        except ValueError:
            _context_to_tag, backoff = obj
        if not _context_to_tag:
            return backoff
        _context_to_tag = {ast.literal_eval(k): v for k, v in _context_to_tag.items()}
        if 'NgramTagger' in cls.__name__:
            return cls(_n, model=_context_to_tag, backoff=backoff)
        else:
            return cls(model=_context_to_tag, backoff=backoff)

    def context(self, tokens, index, history):
        tag_context = tuple(history[max(0, index - self._n + 1):index])
        return (tag_context, tokens[index])