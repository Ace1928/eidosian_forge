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
class Synset(_WordNetObject):
    """Create a Synset from a "<lemma>.<pos>.<number>" string where:
    <lemma> is the word's morphological stem
    <pos> is one of the module attributes ADJ, ADJ_SAT, ADV, NOUN or VERB
    <number> is the sense number, counting from 0.

    Synset attributes, accessible via methods with the same name:

    - name: The canonical name of this synset, formed using the first lemma
      of this synset. Note that this may be different from the name
      passed to the constructor if that string used a different lemma to
      identify the synset.
    - pos: The synset's part of speech, matching one of the module level
      attributes ADJ, ADJ_SAT, ADV, NOUN or VERB.
    - lemmas: A list of the Lemma objects for this synset.
    - definition: The definition for this synset.
    - examples: A list of example strings for this synset.
    - offset: The offset in the WordNet dict file of this synset.
    - lexname: The name of the lexicographer file containing this synset.

    Synset methods:

    Synsets have the following methods for retrieving related Synsets.
    They correspond to the names for the pointer symbols defined here:
    https://wordnet.princeton.edu/documentation/wninput5wn
    These methods all return lists of Synsets.

    - hypernyms, instance_hypernyms
    - hyponyms, instance_hyponyms
    - member_holonyms, substance_holonyms, part_holonyms
    - member_meronyms, substance_meronyms, part_meronyms
    - attributes
    - entailments
    - causes
    - also_sees
    - verb_groups
    - similar_tos

    Additionally, Synsets support the following methods specific to the
    hypernym relation:

    - root_hypernyms
    - common_hypernyms
    - lowest_common_hypernyms

    Note that Synsets do not support the following relations because
    these are defined by WordNet as lexical relations:

    - antonyms
    - derivationally_related_forms
    - pertainyms
    """
    __slots__ = ['_pos', '_offset', '_name', '_frame_ids', '_lemmas', '_lemma_names', '_definition', '_examples', '_lexname', '_pointers', '_lemma_pointers', '_max_depth', '_min_depth']

    def __init__(self, wordnet_corpus_reader):
        self._wordnet_corpus_reader = wordnet_corpus_reader
        self._pos = None
        self._offset = None
        self._name = None
        self._frame_ids = []
        self._lemmas = []
        self._lemma_names = []
        self._definition = None
        self._examples = []
        self._lexname = None
        self._all_hypernyms = None
        self._pointers = defaultdict(set)
        self._lemma_pointers = defaultdict(list)

    def pos(self):
        return self._pos

    def offset(self):
        return self._offset

    def name(self):
        return self._name

    def frame_ids(self):
        return self._frame_ids

    def _doc(self, doc_type, default, lang='eng'):
        """Helper method for Synset.definition and Synset.examples"""
        corpus = self._wordnet_corpus_reader
        if lang not in corpus.langs():
            return None
        elif lang == 'eng':
            return default
        else:
            corpus._load_lang_data(lang)
            of = corpus.ss2of(self)
            i = corpus.lg_attrs.index(doc_type)
            if of in corpus._lang_data[lang][i]:
                return corpus._lang_data[lang][i][of]
            else:
                return None

    def definition(self, lang='eng'):
        """Return definition in specified language"""
        return self._doc('def', self._definition, lang=lang)

    def examples(self, lang='eng'):
        """Return examples in specified language"""
        return self._doc('exe', self._examples, lang=lang)

    def lexname(self):
        return self._lexname

    def _needs_root(self):
        if self._pos == NOUN and self._wordnet_corpus_reader.get_version() != '1.6':
            return False
        else:
            return True

    def lemma_names(self, lang='eng'):
        """Return all the lemma_names associated with the synset"""
        if lang == 'eng':
            return self._lemma_names
        else:
            reader = self._wordnet_corpus_reader
            reader._load_lang_data(lang)
            i = reader.ss2of(self)
            if i in reader._lang_data[lang][0]:
                return reader._lang_data[lang][0][i]
            else:
                return []

    def lemmas(self, lang='eng'):
        """Return all the lemma objects associated with the synset"""
        if lang == 'eng':
            return self._lemmas
        elif self._name:
            self._wordnet_corpus_reader._load_lang_data(lang)
            lemmark = []
            lemmy = self.lemma_names(lang)
            for lem in lemmy:
                temp = Lemma(self._wordnet_corpus_reader, self, lem, self._wordnet_corpus_reader._lexnames.index(self.lexname()), 0, None)
                temp._lang = lang
                lemmark.append(temp)
            return lemmark

    def root_hypernyms(self):
        """Get the topmost hypernyms of this synset in WordNet."""
        result = []
        seen = set()
        todo = [self]
        while todo:
            next_synset = todo.pop()
            if next_synset not in seen:
                seen.add(next_synset)
                next_hypernyms = next_synset.hypernyms() + next_synset.instance_hypernyms()
                if not next_hypernyms:
                    result.append(next_synset)
                else:
                    todo.extend(next_hypernyms)
        return result

    def max_depth(self):
        """
        :return: The length of the longest hypernym path from this
            synset to the root.
        """
        if '_max_depth' not in self.__dict__:
            hypernyms = self.hypernyms() + self.instance_hypernyms()
            if not hypernyms:
                self._max_depth = 0
            else:
                self._max_depth = 1 + max((h.max_depth() for h in hypernyms))
        return self._max_depth

    def min_depth(self):
        """
        :return: The length of the shortest hypernym path from this
            synset to the root.
        """
        if '_min_depth' not in self.__dict__:
            hypernyms = self.hypernyms() + self.instance_hypernyms()
            if not hypernyms:
                self._min_depth = 0
            else:
                self._min_depth = 1 + min((h.min_depth() for h in hypernyms))
        return self._min_depth

    def closure(self, rel, depth=-1):
        """
        Return the transitive closure of source under the rel
        relationship, breadth-first, discarding cycles:

        >>> from nltk.corpus import wordnet as wn
        >>> computer = wn.synset('computer.n.01')
        >>> topic = lambda s:s.topic_domains()
        >>> print(list(computer.closure(topic)))
        [Synset('computer_science.n.01')]

        UserWarning: Discarded redundant search for Synset('computer.n.01') at depth 2


        Include redundant paths (but only once), avoiding duplicate searches
        (from 'animal.n.01' to 'entity.n.01'):

        >>> dog = wn.synset('dog.n.01')
        >>> hyp = lambda s:s.hypernyms()
        >>> print(list(dog.closure(hyp)))
        [Synset('canine.n.02'), Synset('domestic_animal.n.01'), Synset('carnivore.n.01'), Synset('animal.n.01'), Synset('placental.n.01'), Synset('organism.n.01'), Synset('mammal.n.01'), Synset('living_thing.n.01'), Synset('vertebrate.n.01'), Synset('whole.n.02'), Synset('chordate.n.01'), Synset('object.n.01'), Synset('physical_entity.n.01'), Synset('entity.n.01')]

        UserWarning: Discarded redundant search for Synset('animal.n.01') at depth 7
        """
        from nltk.util import acyclic_breadth_first
        for synset in acyclic_breadth_first(self, rel, depth):
            if synset != self:
                yield synset
    from nltk.util import acyclic_depth_first as acyclic_tree
    from nltk.util import unweighted_minimum_spanning_tree as mst

    def tree(self, rel, depth=-1, cut_mark=None):
        """
        Return the full relation tree, including self,
        discarding cycles:

        >>> from nltk.corpus import wordnet as wn
        >>> from pprint import pprint
        >>> computer = wn.synset('computer.n.01')
        >>> topic = lambda s:s.topic_domains()
        >>> pprint(computer.tree(topic))
        [Synset('computer.n.01'), [Synset('computer_science.n.01')]]

        UserWarning: Discarded redundant search for Synset('computer.n.01') at depth -3


        But keep duplicate branches (from 'animal.n.01' to 'entity.n.01'):

        >>> dog = wn.synset('dog.n.01')
        >>> hyp = lambda s:s.hypernyms()
        >>> pprint(dog.tree(hyp))
        [Synset('dog.n.01'),
         [Synset('canine.n.02'),
          [Synset('carnivore.n.01'),
           [Synset('placental.n.01'),
            [Synset('mammal.n.01'),
             [Synset('vertebrate.n.01'),
              [Synset('chordate.n.01'),
               [Synset('animal.n.01'),
                [Synset('organism.n.01'),
                 [Synset('living_thing.n.01'),
                  [Synset('whole.n.02'),
                   [Synset('object.n.01'),
                    [Synset('physical_entity.n.01'),
                     [Synset('entity.n.01')]]]]]]]]]]]]],
         [Synset('domestic_animal.n.01'),
          [Synset('animal.n.01'),
           [Synset('organism.n.01'),
            [Synset('living_thing.n.01'),
             [Synset('whole.n.02'),
              [Synset('object.n.01'),
               [Synset('physical_entity.n.01'), [Synset('entity.n.01')]]]]]]]]]
        """
        from nltk.util import acyclic_branches_depth_first
        return acyclic_branches_depth_first(self, rel, depth, cut_mark)

    def hypernym_paths(self):
        """
        Get the path(s) from this synset to the root, where each path is a
        list of the synset nodes traversed on the way to the root.

        :return: A list of lists, where each list gives the node sequence
           connecting the initial ``Synset`` node and a root node.
        """
        paths = []
        hypernyms = self.hypernyms() + self.instance_hypernyms()
        if len(hypernyms) == 0:
            paths = [[self]]
        for hypernym in hypernyms:
            for ancestor_list in hypernym.hypernym_paths():
                ancestor_list.append(self)
                paths.append(ancestor_list)
        return paths

    def common_hypernyms(self, other):
        """
        Find all synsets that are hypernyms of this synset and the
        other synset.

        :type other: Synset
        :param other: other input synset.
        :return: The synsets that are hypernyms of both synsets.
        """
        if not self._all_hypernyms:
            self._all_hypernyms = {self_synset for self_synsets in self._iter_hypernym_lists() for self_synset in self_synsets}
        if not other._all_hypernyms:
            other._all_hypernyms = {other_synset for other_synsets in other._iter_hypernym_lists() for other_synset in other_synsets}
        return list(self._all_hypernyms.intersection(other._all_hypernyms))

    def lowest_common_hypernyms(self, other, simulate_root=False, use_min_depth=False):
        """
        Get a list of lowest synset(s) that both synsets have as a hypernym.
        When `use_min_depth == False` this means that the synset which appears
        as a hypernym of both `self` and `other` with the lowest maximum depth
        is returned or if there are multiple such synsets at the same depth
        they are all returned

        However, if `use_min_depth == True` then the synset(s) which has/have
        the lowest minimum depth and appear(s) in both paths is/are returned.

        By setting the use_min_depth flag to True, the behavior of NLTK2 can be
        preserved. This was changed in NLTK3 to give more accurate results in a
        small set of cases, generally with synsets concerning people. (eg:
        'chef.n.01', 'fireman.n.01', etc.)

        This method is an implementation of Ted Pedersen's "Lowest Common
        Subsumer" method from the Perl Wordnet module. It can return either
        "self" or "other" if they are a hypernym of the other.

        :type other: Synset
        :param other: other input synset
        :type simulate_root: bool
        :param simulate_root: The various verb taxonomies do not
            share a single root which disallows this metric from working for
            synsets that are not connected. This flag (False by default)
            creates a fake root that connects all the taxonomies. Set it
            to True to enable this behavior. For the noun taxonomy,
            there is usually a default root except for WordNet version 1.6.
            If you are using wordnet 1.6, a fake root will need to be added
            for nouns as well.
        :type use_min_depth: bool
        :param use_min_depth: This setting mimics older (v2) behavior of NLTK
            wordnet If True, will use the min_depth function to calculate the
            lowest common hypernyms. This is known to give strange results for
            some synset pairs (eg: 'chef.n.01', 'fireman.n.01') but is retained
            for backwards compatibility
        :return: The synsets that are the lowest common hypernyms of both
            synsets
        """
        synsets = self.common_hypernyms(other)
        if simulate_root:
            fake_synset = Synset(None)
            fake_synset._name = '*ROOT*'
            fake_synset.hypernyms = lambda: []
            fake_synset.instance_hypernyms = lambda: []
            synsets.append(fake_synset)
        try:
            if use_min_depth:
                max_depth = max((s.min_depth() for s in synsets))
                unsorted_lch = [s for s in synsets if s.min_depth() == max_depth]
            else:
                max_depth = max((s.max_depth() for s in synsets))
                unsorted_lch = [s for s in synsets if s.max_depth() == max_depth]
            return sorted(unsorted_lch)
        except ValueError:
            return []

    def hypernym_distances(self, distance=0, simulate_root=False):
        """
        Get the path(s) from this synset to the root, counting the distance
        of each node from the initial node on the way. A set of
        (synset, distance) tuples is returned.

        :type distance: int
        :param distance: the distance (number of edges) from this hypernym to
            the original hypernym ``Synset`` on which this method was called.
        :return: A set of ``(Synset, int)`` tuples where each ``Synset`` is
           a hypernym of the first ``Synset``.
        """
        distances = {(self, distance)}
        for hypernym in self._hypernyms() + self._instance_hypernyms():
            distances |= hypernym.hypernym_distances(distance + 1, simulate_root=False)
        if simulate_root:
            fake_synset = Synset(None)
            fake_synset._name = '*ROOT*'
            fake_synset_distance = max(distances, key=itemgetter(1))[1]
            distances.add((fake_synset, fake_synset_distance + 1))
        return distances

    def _shortest_hypernym_paths(self, simulate_root):
        if self._name == '*ROOT*':
            return {self: 0}
        queue = deque([(self, 0)])
        path = {}
        while queue:
            s, depth = queue.popleft()
            if s in path:
                continue
            path[s] = depth
            depth += 1
            queue.extend(((hyp, depth) for hyp in s._hypernyms()))
            queue.extend(((hyp, depth) for hyp in s._instance_hypernyms()))
        if simulate_root:
            fake_synset = Synset(None)
            fake_synset._name = '*ROOT*'
            path[fake_synset] = max(path.values()) + 1
        return path

    def shortest_path_distance(self, other, simulate_root=False):
        """
        Returns the distance of the shortest path linking the two synsets (if
        one exists). For each synset, all the ancestor nodes and their
        distances are recorded and compared. The ancestor node common to both
        synsets that can be reached with the minimum number of traversals is
        used. If no ancestor nodes are common, None is returned. If a node is
        compared with itself 0 is returned.

        :type other: Synset
        :param other: The Synset to which the shortest path will be found.
        :return: The number of edges in the shortest path connecting the two
            nodes, or None if no path exists.
        """
        if self == other:
            return 0
        dist_dict1 = self._shortest_hypernym_paths(simulate_root)
        dist_dict2 = other._shortest_hypernym_paths(simulate_root)
        inf = float('inf')
        path_distance = inf
        for synset, d1 in dist_dict1.items():
            d2 = dist_dict2.get(synset, inf)
            path_distance = min(path_distance, d1 + d2)
        return None if math.isinf(path_distance) else path_distance

    def path_similarity(self, other, verbose=False, simulate_root=True):
        """
        Path Distance Similarity:
        Return a score denoting how similar two word senses are, based on the
        shortest path that connects the senses in the is-a (hypernym/hypnoym)
        taxonomy. The score is in the range 0 to 1, except in those cases where
        a path cannot be found (will only be true for verbs as there are many
        distinct verb taxonomies), in which case None is returned. A score of
        1 represents identity i.e. comparing a sense with itself will return 1.

        :type other: Synset
        :param other: The ``Synset`` that this ``Synset`` is being compared to.
        :type simulate_root: bool
        :param simulate_root: The various verb taxonomies do not
            share a single root which disallows this metric from working for
            synsets that are not connected. This flag (True by default)
            creates a fake root that connects all the taxonomies. Set it
            to false to disable this behavior. For the noun taxonomy,
            there is usually a default root except for WordNet version 1.6.
            If you are using wordnet 1.6, a fake root will be added for nouns
            as well.
        :return: A score denoting the similarity of the two ``Synset`` objects,
            normally between 0 and 1. None is returned if no connecting path
            could be found. 1 is returned if a ``Synset`` is compared with
            itself.
        """
        distance = self.shortest_path_distance(other, simulate_root=simulate_root and (self._needs_root() or other._needs_root()))
        if distance is None or distance < 0:
            return None
        return 1.0 / (distance + 1)

    def lch_similarity(self, other, verbose=False, simulate_root=True):
        """
        Leacock Chodorow Similarity:
        Return a score denoting how similar two word senses are, based on the
        shortest path that connects the senses (as above) and the maximum depth
        of the taxonomy in which the senses occur. The relationship is given as
        -log(p/2d) where p is the shortest path length and d is the taxonomy
        depth.

        :type  other: Synset
        :param other: The ``Synset`` that this ``Synset`` is being compared to.
        :type simulate_root: bool
        :param simulate_root: The various verb taxonomies do not
            share a single root which disallows this metric from working for
            synsets that are not connected. This flag (True by default)
            creates a fake root that connects all the taxonomies. Set it
            to false to disable this behavior. For the noun taxonomy,
            there is usually a default root except for WordNet version 1.6.
            If you are using wordnet 1.6, a fake root will be added for nouns
            as well.
        :return: A score denoting the similarity of the two ``Synset`` objects,
            normally greater than 0. None is returned if no connecting path
            could be found. If a ``Synset`` is compared with itself, the
            maximum score is returned, which varies depending on the taxonomy
            depth.
        """
        if self._pos != other._pos:
            raise WordNetError('Computing the lch similarity requires %s and %s to have the same part of speech.' % (self, other))
        need_root = self._needs_root()
        if self._pos not in self._wordnet_corpus_reader._max_depth:
            self._wordnet_corpus_reader._compute_max_depth(self._pos, need_root)
        depth = self._wordnet_corpus_reader._max_depth[self._pos]
        distance = self.shortest_path_distance(other, simulate_root=simulate_root and need_root)
        if distance is None or distance < 0 or depth == 0:
            return None
        return -math.log((distance + 1) / (2.0 * depth))

    def wup_similarity(self, other, verbose=False, simulate_root=True):
        """
        Wu-Palmer Similarity:
        Return a score denoting how similar two word senses are, based on the
        depth of the two senses in the taxonomy and that of their Least Common
        Subsumer (most specific ancestor node). Previously, the scores computed
        by this implementation did _not_ always agree with those given by
        Pedersen's Perl implementation of WordNet Similarity. However, with
        the addition of the simulate_root flag (see below), the score for
        verbs now almost always agree but not always for nouns.

        The LCS does not necessarily feature in the shortest path connecting
        the two senses, as it is by definition the common ancestor deepest in
        the taxonomy, not closest to the two senses. Typically, however, it
        will so feature. Where multiple candidates for the LCS exist, that
        whose shortest path to the root node is the longest will be selected.
        Where the LCS has multiple paths to the root, the longer path is used
        for the purposes of the calculation.

        :type  other: Synset
        :param other: The ``Synset`` that this ``Synset`` is being compared to.
        :type simulate_root: bool
        :param simulate_root: The various verb taxonomies do not
            share a single root which disallows this metric from working for
            synsets that are not connected. This flag (True by default)
            creates a fake root that connects all the taxonomies. Set it
            to false to disable this behavior. For the noun taxonomy,
            there is usually a default root except for WordNet version 1.6.
            If you are using wordnet 1.6, a fake root will be added for nouns
            as well.
        :return: A float score denoting the similarity of the two ``Synset``
            objects, normally greater than zero. If no connecting path between
            the two senses can be found, None is returned.

        """
        need_root = self._needs_root() or other._needs_root()
        subsumers = self.lowest_common_hypernyms(other, simulate_root=simulate_root and need_root, use_min_depth=True)
        if len(subsumers) == 0:
            return None
        subsumer = self if self in subsumers else subsumers[0]
        depth = subsumer.max_depth() + 1
        len1 = self.shortest_path_distance(subsumer, simulate_root=simulate_root and need_root)
        len2 = other.shortest_path_distance(subsumer, simulate_root=simulate_root and need_root)
        if len1 is None or len2 is None:
            return None
        len1 += depth
        len2 += depth
        return 2.0 * depth / (len1 + len2)

    def res_similarity(self, other, ic, verbose=False):
        """
        Resnik Similarity:
        Return a score denoting how similar two word senses are, based on the
        Information Content (IC) of the Least Common Subsumer (most specific
        ancestor node).

        :type  other: Synset
        :param other: The ``Synset`` that this ``Synset`` is being compared to.
        :type ic: dict
        :param ic: an information content object (as returned by
            ``nltk.corpus.wordnet_ic.ic()``).
        :return: A float score denoting the similarity of the two ``Synset``
            objects. Synsets whose LCS is the root node of the taxonomy will
            have a score of 0 (e.g. N['dog'][0] and N['table'][0]).
        """
        ic1, ic2, lcs_ic = _lcs_ic(self, other, ic)
        return lcs_ic

    def jcn_similarity(self, other, ic, verbose=False):
        """
        Jiang-Conrath Similarity:
        Return a score denoting how similar two word senses are, based on the
        Information Content (IC) of the Least Common Subsumer (most specific
        ancestor node) and that of the two input Synsets. The relationship is
        given by the equation 1 / (IC(s1) + IC(s2) - 2 * IC(lcs)).

        :type  other: Synset
        :param other: The ``Synset`` that this ``Synset`` is being compared to.
        :type  ic: dict
        :param ic: an information content object (as returned by
            ``nltk.corpus.wordnet_ic.ic()``).
        :return: A float score denoting the similarity of the two ``Synset``
            objects.
        """
        if self == other:
            return _INF
        ic1, ic2, lcs_ic = _lcs_ic(self, other, ic)
        if ic1 == 0 or ic2 == 0:
            return 0
        ic_difference = ic1 + ic2 - 2 * lcs_ic
        if ic_difference == 0:
            return _INF
        return 1 / ic_difference

    def lin_similarity(self, other, ic, verbose=False):
        """
        Lin Similarity:
        Return a score denoting how similar two word senses are, based on the
        Information Content (IC) of the Least Common Subsumer (most specific
        ancestor node) and that of the two input Synsets. The relationship is
        given by the equation 2 * IC(lcs) / (IC(s1) + IC(s2)).

        :type other: Synset
        :param other: The ``Synset`` that this ``Synset`` is being compared to.
        :type ic: dict
        :param ic: an information content object (as returned by
            ``nltk.corpus.wordnet_ic.ic()``).
        :return: A float score denoting the similarity of the two ``Synset``
            objects, in the range 0 to 1.
        """
        ic1, ic2, lcs_ic = _lcs_ic(self, other, ic)
        return 2.0 * lcs_ic / (ic1 + ic2)

    def _iter_hypernym_lists(self):
        """
        :return: An iterator over ``Synset`` objects that are either proper
        hypernyms or instance of hypernyms of the synset.
        """
        todo = [self]
        seen = set()
        while todo:
            for synset in todo:
                seen.add(synset)
            yield todo
            todo = [hypernym for synset in todo for hypernym in synset.hypernyms() + synset.instance_hypernyms() if hypernym not in seen]

    def __repr__(self):
        return f"{type(self).__name__}('{self._name}')"

    def _related(self, relation_symbol, sort=True):
        get_synset = self._wordnet_corpus_reader.synset_from_pos_and_offset
        if relation_symbol not in self._pointers:
            return []
        pointer_tuples = self._pointers[relation_symbol]
        r = [get_synset(pos, offset) for pos, offset in pointer_tuples]
        if sort:
            r.sort()
        return r