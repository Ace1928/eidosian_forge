from time import perf_counter
from nltk.featstruct import TYPE, FeatStruct, find_variables, unify
from nltk.grammar import (
from nltk.parse.chart import (
from nltk.sem import logic
from nltk.tree import Tree
class FeatureBottomUpPredictCombineRule(BottomUpPredictCombineRule):

    def apply(self, chart, grammar, edge):
        if edge.is_incomplete():
            return
        found = edge.lhs()
        for prod in grammar.productions(rhs=found):
            bindings = {}
            if isinstance(edge, FeatureTreeEdge):
                _next = prod.rhs()[0]
                if not is_nonterminal(_next):
                    continue
                used_vars = find_variables((prod.lhs(),) + prod.rhs(), fs_class=FeatStruct)
                found = found.rename_variables(used_vars=used_vars)
                result = unify(_next, found, bindings, rename_vars=False)
                if result is None:
                    continue
            new_edge = FeatureTreeEdge.from_production(prod, edge.start()).move_dot_forward(edge.end(), bindings)
            if chart.insert(new_edge, (edge,)):
                yield new_edge