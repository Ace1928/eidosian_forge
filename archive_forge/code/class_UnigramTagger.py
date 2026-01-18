import ast
import re
from abc import abstractmethod
from typing import List, Optional, Tuple
from nltk import jsontags
from nltk.classify import NaiveBayesClassifier
from nltk.probability import ConditionalFreqDist
from nltk.tag.api import FeaturesetTaggerI, TaggerI
@jsontags.register_tag
class UnigramTagger(NgramTagger):
    """
    Unigram Tagger

    The UnigramTagger finds the most likely tag for each word in a training
    corpus, and then uses that information to assign tags to new tokens.

        >>> from nltk.corpus import brown
        >>> from nltk.tag import UnigramTagger
        >>> test_sent = brown.sents(categories='news')[0]
        >>> unigram_tagger = UnigramTagger(brown.tagged_sents(categories='news')[:500])
        >>> for tok, tag in unigram_tagger.tag(test_sent):
        ...     print("({}, {}), ".format(tok, tag)) # doctest: +NORMALIZE_WHITESPACE
        (The, AT), (Fulton, NP-TL), (County, NN-TL), (Grand, JJ-TL),
        (Jury, NN-TL), (said, VBD), (Friday, NR), (an, AT),
        (investigation, NN), (of, IN), (Atlanta's, NP$), (recent, JJ),
        (primary, NN), (election, NN), (produced, VBD), (``, ``),
        (no, AT), (evidence, NN), ('', ''), (that, CS), (any, DTI),
        (irregularities, NNS), (took, VBD), (place, NN), (., .),

    :param train: The corpus of training data, a list of tagged sentences
    :type train: list(list(tuple(str, str)))
    :param model: The tagger model
    :type model: dict
    :param backoff: Another tagger which this tagger will consult when it is
        unable to tag a word
    :type backoff: TaggerI
    :param cutoff: The number of instances of training data the tagger must see
        in order not to use the backoff tagger
    :type cutoff: int
    """
    json_tag = 'nltk.tag.sequential.UnigramTagger'

    def __init__(self, train=None, model=None, backoff=None, cutoff=0, verbose=False):
        super().__init__(1, train, model, backoff, cutoff, verbose)

    def context(self, tokens, index, history):
        return tokens[index]