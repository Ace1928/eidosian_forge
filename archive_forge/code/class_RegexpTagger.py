import ast
import re
from abc import abstractmethod
from typing import List, Optional, Tuple
from nltk import jsontags
from nltk.classify import NaiveBayesClassifier
from nltk.probability import ConditionalFreqDist
from nltk.tag.api import FeaturesetTaggerI, TaggerI
@jsontags.register_tag
class RegexpTagger(SequentialBackoffTagger):
    """
    Regular Expression Tagger

    The RegexpTagger assigns tags to tokens by comparing their
    word strings to a series of regular expressions.  The following tagger
    uses word suffixes to make guesses about the correct Brown Corpus part
    of speech tag:

        >>> from nltk.corpus import brown
        >>> from nltk.tag import RegexpTagger
        >>> test_sent = brown.sents(categories='news')[0]
        >>> regexp_tagger = RegexpTagger(
        ...     [(r'^-?[0-9]+(\\.[0-9]+)?$', 'CD'),  # cardinal numbers
        ...      (r'(The|the|A|a|An|an)$', 'AT'),   # articles
        ...      (r'.*able$', 'JJ'),                # adjectives
        ...      (r'.*ness$', 'NN'),                # nouns formed from adjectives
        ...      (r'.*ly$', 'RB'),                  # adverbs
        ...      (r'.*s$', 'NNS'),                  # plural nouns
        ...      (r'.*ing$', 'VBG'),                # gerunds
        ...      (r'.*ed$', 'VBD'),                 # past tense verbs
        ...      (r'.*', 'NN')                      # nouns (default)
        ... ])
        >>> regexp_tagger
        <Regexp Tagger: size=9>
        >>> regexp_tagger.tag(test_sent) # doctest: +NORMALIZE_WHITESPACE
        [('The', 'AT'), ('Fulton', 'NN'), ('County', 'NN'), ('Grand', 'NN'), ('Jury', 'NN'),
        ('said', 'NN'), ('Friday', 'NN'), ('an', 'AT'), ('investigation', 'NN'), ('of', 'NN'),
        ("Atlanta's", 'NNS'), ('recent', 'NN'), ('primary', 'NN'), ('election', 'NN'),
        ('produced', 'VBD'), ('``', 'NN'), ('no', 'NN'), ('evidence', 'NN'), ("''", 'NN'),
        ('that', 'NN'), ('any', 'NN'), ('irregularities', 'NNS'), ('took', 'NN'),
        ('place', 'NN'), ('.', 'NN')]

    :type regexps: list(tuple(str, str))
    :param regexps: A list of ``(regexp, tag)`` pairs, each of
        which indicates that a word matching ``regexp`` should
        be tagged with ``tag``.  The pairs will be evaluated in
        order.  If none of the regexps match a word, then the
        optional backoff tagger is invoked, else it is
        assigned the tag None.
    """
    json_tag = 'nltk.tag.sequential.RegexpTagger'

    def __init__(self, regexps: List[Tuple[str, str]], backoff: Optional[TaggerI]=None):
        super().__init__(backoff)
        self._regexps = []
        for regexp, tag in regexps:
            try:
                self._regexps.append((re.compile(regexp), tag))
            except Exception as e:
                raise Exception(f'Invalid RegexpTagger regexp: {e}\n- regexp: {regexp!r}\n- tag: {tag!r}') from e

    def encode_json_obj(self):
        return ([(regexp.pattern, tag) for regexp, tag in self._regexps], self.backoff)

    @classmethod
    def decode_json_obj(cls, obj):
        regexps, backoff = obj
        return cls(regexps, backoff)

    def choose_tag(self, tokens, index, history):
        for regexp, tag in self._regexps:
            if re.match(regexp, tokens[index]):
                return tag
        return None

    def __repr__(self):
        return f'<Regexp Tagger: size={len(self._regexps)}>'