from collections import defaultdict
from functools import total_ordering
from itertools import chain
from nltk.grammar import (
from nltk.internals import raise_unorderable_types
from nltk.parse.dependencygraph import DependencyGraph
class ProbabilisticProjectiveDependencyParser:
    """A probabilistic, projective dependency parser.

    This parser returns the most probable projective parse derived from the
    probabilistic dependency grammar derived from the train() method.  The
    probabilistic model is an implementation of Eisner's (1996) Model C, which
    conditions on head-word, head-tag, child-word, and child-tag.  The decoding
    uses a bottom-up chart-based span concatenation algorithm that's identical
    to the one utilized by the rule-based projective parser.

    Usage example

    >>> from nltk.parse.dependencygraph import conll_data2

    >>> graphs = [
    ... DependencyGraph(entry) for entry in conll_data2.split('\\n\\n') if entry
    ... ]

    >>> ppdp = ProbabilisticProjectiveDependencyParser()
    >>> ppdp.train(graphs)

    >>> sent = ['Cathy', 'zag', 'hen', 'wild', 'zwaaien', '.']
    >>> list(ppdp.parse(sent))
    [Tree('zag', ['Cathy', 'hen', Tree('zwaaien', ['wild', '.'])])]

    """

    def __init__(self):
        """
        Create a new probabilistic dependency parser.  No additional
        operations are necessary.
        """

    def parse(self, tokens):
        """
        Parses the list of tokens subject to the projectivity constraint
        and the productions in the parser's grammar.  This uses a method
        similar to the span-concatenation algorithm defined in Eisner (1996).
        It returns the most probable parse derived from the parser's
        probabilistic dependency grammar.
        """
        self._tokens = list(tokens)
        chart = []
        for i in range(0, len(self._tokens) + 1):
            chart.append([])
            for j in range(0, len(self._tokens) + 1):
                chart[i].append(ChartCell(i, j))
                if i == j + 1:
                    if tokens[i - 1] in self._grammar._tags:
                        for tag in self._grammar._tags[tokens[i - 1]]:
                            chart[i][j].add(DependencySpan(i - 1, i, i - 1, [-1], [tag]))
                    else:
                        print("No tag found for input token '%s', parse is impossible." % tokens[i - 1])
                        return []
        for i in range(1, len(self._tokens) + 1):
            for j in range(i - 2, -1, -1):
                for k in range(i - 1, j, -1):
                    for span1 in chart[k][j]._entries:
                        for span2 in chart[i][k]._entries:
                            for newspan in self.concatenate(span1, span2):
                                chart[i][j].add(newspan)
        trees = []
        max_parse = None
        max_score = 0
        for parse in chart[len(self._tokens)][0]._entries:
            conll_format = ''
            malt_format = ''
            for i in range(len(tokens)):
                malt_format += '%s\t%s\t%d\t%s\n' % (tokens[i], 'null', parse._arcs[i] + 1, 'null')
                conll_format += '\t%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n' % (i + 1, tokens[i], tokens[i], parse._tags[i], parse._tags[i], 'null', parse._arcs[i] + 1, 'ROOT', '-', '-')
            dg = DependencyGraph(conll_format)
            score = self.compute_prob(dg)
            trees.append((score, dg.tree()))
        trees.sort()
        return (tree for score, tree in trees)

    def concatenate(self, span1, span2):
        """
        Concatenates the two spans in whichever way possible.  This
        includes rightward concatenation (from the leftmost word of the
        leftmost span to the rightmost word of the rightmost span) and
        leftward concatenation (vice-versa) between adjacent spans.  Unlike
        Eisner's presentation of span concatenation, these spans do not
        share or pivot on a particular word/word-index.

        :return: A list of new spans formed through concatenation.
        :rtype: list(DependencySpan)
        """
        spans = []
        if span1._start_index == span2._start_index:
            print('Error: Mismatched spans - replace this with thrown error')
        if span1._start_index > span2._start_index:
            temp_span = span1
            span1 = span2
            span2 = temp_span
        new_arcs = span1._arcs + span2._arcs
        new_tags = span1._tags + span2._tags
        if self._grammar.contains(self._tokens[span1._head_index], self._tokens[span2._head_index]):
            new_arcs[span2._head_index - span1._start_index] = span1._head_index
            spans.append(DependencySpan(span1._start_index, span2._end_index, span1._head_index, new_arcs, new_tags))
        new_arcs = span1._arcs + span2._arcs
        new_tags = span1._tags + span2._tags
        if self._grammar.contains(self._tokens[span2._head_index], self._tokens[span1._head_index]):
            new_arcs[span1._head_index - span1._start_index] = span2._head_index
            spans.append(DependencySpan(span1._start_index, span2._end_index, span2._head_index, new_arcs, new_tags))
        return spans

    def train(self, graphs):
        """
        Trains a ProbabilisticDependencyGrammar based on the list of input
        DependencyGraphs.  This model is an implementation of Eisner's (1996)
        Model C, which derives its statistics from head-word, head-tag,
        child-word, and child-tag relationships.

        :param graphs: A list of dependency graphs to train from.
        :type: list(DependencyGraph)
        """
        productions = []
        events = defaultdict(int)
        tags = {}
        for dg in graphs:
            for node_index in range(1, len(dg.nodes)):
                children = list(chain.from_iterable(dg.nodes[node_index]['deps'].values()))
                nr_left_children = dg.left_children(node_index)
                nr_right_children = dg.right_children(node_index)
                nr_children = nr_left_children + nr_right_children
                for child_index in range(0 - (nr_left_children + 1), nr_right_children + 2):
                    head_word = dg.nodes[node_index]['word']
                    head_tag = dg.nodes[node_index]['tag']
                    if head_word in tags:
                        tags[head_word].add(head_tag)
                    else:
                        tags[head_word] = {head_tag}
                    child = 'STOP'
                    child_tag = 'STOP'
                    prev_word = 'START'
                    prev_tag = 'START'
                    if child_index < 0:
                        array_index = child_index + nr_left_children
                        if array_index >= 0:
                            child = dg.nodes[children[array_index]]['word']
                            child_tag = dg.nodes[children[array_index]]['tag']
                        if child_index != -1:
                            prev_word = dg.nodes[children[array_index + 1]]['word']
                            prev_tag = dg.nodes[children[array_index + 1]]['tag']
                        if child != 'STOP':
                            productions.append(DependencyProduction(head_word, [child]))
                        head_event = '(head ({} {}) (mods ({}, {}, {}) left))'.format(child, child_tag, prev_tag, head_word, head_tag)
                        mod_event = '(mods ({}, {}, {}) left))'.format(prev_tag, head_word, head_tag)
                        events[head_event] += 1
                        events[mod_event] += 1
                    elif child_index > 0:
                        array_index = child_index + nr_left_children - 1
                        if array_index < nr_children:
                            child = dg.nodes[children[array_index]]['word']
                            child_tag = dg.nodes[children[array_index]]['tag']
                        if child_index != 1:
                            prev_word = dg.nodes[children[array_index - 1]]['word']
                            prev_tag = dg.nodes[children[array_index - 1]]['tag']
                        if child != 'STOP':
                            productions.append(DependencyProduction(head_word, [child]))
                        head_event = '(head ({} {}) (mods ({}, {}, {}) right))'.format(child, child_tag, prev_tag, head_word, head_tag)
                        mod_event = '(mods ({}, {}, {}) right))'.format(prev_tag, head_word, head_tag)
                        events[head_event] += 1
                        events[mod_event] += 1
        self._grammar = ProbabilisticDependencyGrammar(productions, events, tags)

    def compute_prob(self, dg):
        """
        Computes the probability of a dependency graph based
        on the parser's probability model (defined by the parser's
        statistical dependency grammar).

        :param dg: A dependency graph to score.
        :type dg: DependencyGraph
        :return: The probability of the dependency graph.
        :rtype: int
        """
        prob = 1.0
        for node_index in range(1, len(dg.nodes)):
            children = list(chain.from_iterable(dg.nodes[node_index]['deps'].values()))
            nr_left_children = dg.left_children(node_index)
            nr_right_children = dg.right_children(node_index)
            nr_children = nr_left_children + nr_right_children
            for child_index in range(0 - (nr_left_children + 1), nr_right_children + 2):
                head_word = dg.nodes[node_index]['word']
                head_tag = dg.nodes[node_index]['tag']
                child = 'STOP'
                child_tag = 'STOP'
                prev_word = 'START'
                prev_tag = 'START'
                if child_index < 0:
                    array_index = child_index + nr_left_children
                    if array_index >= 0:
                        child = dg.nodes[children[array_index]]['word']
                        child_tag = dg.nodes[children[array_index]]['tag']
                    if child_index != -1:
                        prev_word = dg.nodes[children[array_index + 1]]['word']
                        prev_tag = dg.nodes[children[array_index + 1]]['tag']
                    head_event = '(head ({} {}) (mods ({}, {}, {}) left))'.format(child, child_tag, prev_tag, head_word, head_tag)
                    mod_event = '(mods ({}, {}, {}) left))'.format(prev_tag, head_word, head_tag)
                    h_count = self._grammar._events[head_event]
                    m_count = self._grammar._events[mod_event]
                    if m_count != 0:
                        prob *= h_count / m_count
                    else:
                        prob = 1e-08
                elif child_index > 0:
                    array_index = child_index + nr_left_children - 1
                    if array_index < nr_children:
                        child = dg.nodes[children[array_index]]['word']
                        child_tag = dg.nodes[children[array_index]]['tag']
                    if child_index != 1:
                        prev_word = dg.nodes[children[array_index - 1]]['word']
                        prev_tag = dg.nodes[children[array_index - 1]]['tag']
                    head_event = '(head ({} {}) (mods ({}, {}, {}) right))'.format(child, child_tag, prev_tag, head_word, head_tag)
                    mod_event = '(mods ({}, {}, {}) right))'.format(prev_tag, head_word, head_tag)
                    h_count = self._grammar._events[head_event]
                    m_count = self._grammar._events[mod_event]
                    if m_count != 0:
                        prob *= h_count / m_count
                    else:
                        prob = 1e-08
        return prob