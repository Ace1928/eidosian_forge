import ast
import re
from abc import abstractmethod
from typing import List, Optional, Tuple
from nltk import jsontags
from nltk.classify import NaiveBayesClassifier
from nltk.probability import ConditionalFreqDist
from nltk.tag.api import FeaturesetTaggerI, TaggerI
@jsontags.register_tag
class DefaultTagger(SequentialBackoffTagger):
    """
    A tagger that assigns the same tag to every token.

        >>> from nltk.tag import DefaultTagger
        >>> default_tagger = DefaultTagger('NN')
        >>> list(default_tagger.tag('This is a test'.split()))
        [('This', 'NN'), ('is', 'NN'), ('a', 'NN'), ('test', 'NN')]

    This tagger is recommended as a backoff tagger, in cases where
    a more powerful tagger is unable to assign a tag to the word
    (e.g. because the word was not seen during training).

    :param tag: The tag to assign to each token
    :type tag: str
    """
    json_tag = 'nltk.tag.sequential.DefaultTagger'

    def __init__(self, tag):
        self._tag = tag
        super().__init__(None)

    def encode_json_obj(self):
        return self._tag

    @classmethod
    def decode_json_obj(cls, obj):
        tag = obj
        return cls(tag)

    def choose_tag(self, tokens, index, history):
        return self._tag

    def __repr__(self):
        return f'<DefaultTagger: tag={self._tag}>'