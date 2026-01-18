import ast
import re
from abc import abstractmethod
from typing import List, Optional, Tuple
from nltk import jsontags
from nltk.classify import NaiveBayesClassifier
from nltk.probability import ConditionalFreqDist
from nltk.tag.api import FeaturesetTaggerI, TaggerI
class SequentialBackoffTagger(TaggerI):
    """
    An abstract base class for taggers that tags words sequentially,
    left to right.  Tagging of individual words is performed by the
    ``choose_tag()`` method, which should be defined by subclasses.  If
    a tagger is unable to determine a tag for the specified token,
    then its backoff tagger is consulted.

    :ivar _taggers: A list of all the taggers that should be tried to
        tag a token (i.e., self and its backoff taggers).
    """

    def __init__(self, backoff=None):
        if backoff is None:
            self._taggers = [self]
        else:
            self._taggers = [self] + backoff._taggers

    @property
    def backoff(self):
        """The backoff tagger for this tagger."""
        return self._taggers[1] if len(self._taggers) > 1 else None

    def tag(self, tokens):
        tags = []
        for i in range(len(tokens)):
            tags.append(self.tag_one(tokens, i, tags))
        return list(zip(tokens, tags))

    def tag_one(self, tokens, index, history):
        """
        Determine an appropriate tag for the specified token, and
        return that tag.  If this tagger is unable to determine a tag
        for the specified token, then its backoff tagger is consulted.

        :rtype: str
        :type tokens: list
        :param tokens: The list of words that are being tagged.
        :type index: int
        :param index: The index of the word whose tag should be
            returned.
        :type history: list(str)
        :param history: A list of the tags for all words before *index*.
        """
        tag = None
        for tagger in self._taggers:
            tag = tagger.choose_tag(tokens, index, history)
            if tag is not None:
                break
        return tag

    @abstractmethod
    def choose_tag(self, tokens, index, history):
        """
        Decide which tag should be used for the specified token, and
        return that tag.  If this tagger is unable to determine a tag
        for the specified token, return None -- do not consult
        the backoff tagger.  This method should be overridden by
        subclasses of SequentialBackoffTagger.

        :rtype: str
        :type tokens: list
        :param tokens: The list of words that are being tagged.
        :type index: int
        :param index: The index of the word whose tag should be
            returned.
        :type history: list(str)
        :param history: A list of the tags for all words before *index*.
        """