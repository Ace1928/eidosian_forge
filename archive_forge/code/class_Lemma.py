import math
import os
import re
import warnings
from collections import defaultdict, deque
from functools import total_ordering
from itertools import chain, islice
from operator import itemgetter
from nltk.corpus.reader import CorpusReader
from nltk.internals import deprecated
from nltk.probability import FreqDist
from nltk.util import binary_search_file as _binary_search_file
class Lemma(_WordNetObject):
    """
    The lexical entry for a single morphological form of a
    sense-disambiguated word.

    Create a Lemma from a "<word>.<pos>.<number>.<lemma>" string where:
    <word> is the morphological stem identifying the synset
    <pos> is one of the module attributes ADJ, ADJ_SAT, ADV, NOUN or VERB
    <number> is the sense number, counting from 0.
    <lemma> is the morphological form of interest

    Note that <word> and <lemma> can be different, e.g. the Synset
    'salt.n.03' has the Lemmas 'salt.n.03.salt', 'salt.n.03.saltiness' and
    'salt.n.03.salinity'.

    Lemma attributes, accessible via methods with the same name:

    - name: The canonical name of this lemma.
    - synset: The synset that this lemma belongs to.
    - syntactic_marker: For adjectives, the WordNet string identifying the
      syntactic position relative modified noun. See:
      https://wordnet.princeton.edu/documentation/wninput5wn
      For all other parts of speech, this attribute is None.
    - count: The frequency of this lemma in wordnet.

    Lemma methods:

    Lemmas have the following methods for retrieving related Lemmas. They
    correspond to the names for the pointer symbols defined here:
    https://wordnet.princeton.edu/documentation/wninput5wn
    These methods all return lists of Lemmas:

    - antonyms
    - hypernyms, instance_hypernyms
    - hyponyms, instance_hyponyms
    - member_holonyms, substance_holonyms, part_holonyms
    - member_meronyms, substance_meronyms, part_meronyms
    - topic_domains, region_domains, usage_domains
    - attributes
    - derivationally_related_forms
    - entailments
    - causes
    - also_sees
    - verb_groups
    - similar_tos
    - pertainyms
    """
    __slots__ = ['_wordnet_corpus_reader', '_name', '_syntactic_marker', '_synset', '_frame_strings', '_frame_ids', '_lexname_index', '_lex_id', '_lang', '_key']

    def __init__(self, wordnet_corpus_reader, synset, name, lexname_index, lex_id, syntactic_marker):
        self._wordnet_corpus_reader = wordnet_corpus_reader
        self._name = name
        self._syntactic_marker = syntactic_marker
        self._synset = synset
        self._frame_strings = []
        self._frame_ids = []
        self._lexname_index = lexname_index
        self._lex_id = lex_id
        self._lang = 'eng'
        self._key = None

    def name(self):
        return self._name

    def syntactic_marker(self):
        return self._syntactic_marker

    def synset(self):
        return self._synset

    def frame_strings(self):
        return self._frame_strings

    def frame_ids(self):
        return self._frame_ids

    def lang(self):
        return self._lang

    def key(self):
        return self._key

    def __repr__(self):
        tup = (type(self).__name__, self._synset._name, self._name)
        return "%s('%s.%s')" % tup

    def _related(self, relation_symbol):
        get_synset = self._wordnet_corpus_reader.synset_from_pos_and_offset
        if (self._name, relation_symbol) not in self._synset._lemma_pointers:
            return []
        return [get_synset(pos, offset)._lemmas[lemma_index] for pos, offset, lemma_index in self._synset._lemma_pointers[self._name, relation_symbol]]

    def count(self):
        """Return the frequency count for this Lemma"""
        return self._wordnet_corpus_reader.lemma_count(self)

    def antonyms(self):
        return self._related('!')

    def derivationally_related_forms(self):
        return self._related('+')

    def pertainyms(self):
        return self._related('\\')