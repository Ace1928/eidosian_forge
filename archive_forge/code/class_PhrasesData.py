import logging
import unittest
import numpy as np
from gensim.models.phrases import Phrases, FrozenPhrases, _PhrasesTransformation
from gensim.models.phrases import original_scorer
from gensim.test.utils import common_texts, temporary_file, datapath
class PhrasesData:
    sentences = common_texts + [['graph', 'minors', 'survey', 'human', 'interface']]
    connector_words = frozenset()
    bigram1 = u'response_time'
    bigram2 = u'graph_minors'
    bigram3 = u'human_interface'

    def gen_sentences(self):
        return ((w for w in sentence) for sentence in self.sentences)