from collections.abc import Iterable
from io import StringIO
from numbers import Integral
import numpy as np
from ..base import is_classifier
from ..utils._param_validation import HasMethods, Interval, StrOptions, validate_params
from ..utils.validation import check_array, check_is_fitted
from . import DecisionTreeClassifier, DecisionTreeRegressor, _criterion, _tree
from ._reingold_tilford import Tree, buchheim
class _DOTTreeExporter(_BaseTreeExporter):

    def __init__(self, out_file=SENTINEL, max_depth=None, feature_names=None, class_names=None, label='all', filled=False, leaves_parallel=False, impurity=True, node_ids=False, proportion=False, rotate=False, rounded=False, special_characters=False, precision=3, fontname='helvetica'):
        super().__init__(max_depth=max_depth, feature_names=feature_names, class_names=class_names, label=label, filled=filled, impurity=impurity, node_ids=node_ids, proportion=proportion, rounded=rounded, precision=precision)
        self.leaves_parallel = leaves_parallel
        self.out_file = out_file
        self.special_characters = special_characters
        self.fontname = fontname
        self.rotate = rotate
        if special_characters:
            self.characters = ['&#35;', '<SUB>', '</SUB>', '&le;', '<br/>', '>', '<']
        else:
            self.characters = ['#', '[', ']', '<=', '\\n', '"', '"']
        self.ranks = {'leaves': []}
        self.colors = {'bounds': None}

    def export(self, decision_tree):
        if self.feature_names is not None:
            if len(self.feature_names) != decision_tree.n_features_in_:
                raise ValueError('Length of feature_names, %d does not match number of features, %d' % (len(self.feature_names), decision_tree.n_features_in_))
        self.head()
        if isinstance(decision_tree, _tree.Tree):
            self.recurse(decision_tree, 0, criterion='impurity')
        else:
            self.recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)
        self.tail()

    def tail(self):
        if self.leaves_parallel:
            for rank in sorted(self.ranks):
                self.out_file.write('{rank=same ; ' + '; '.join((r for r in self.ranks[rank])) + '} ;\n')
        self.out_file.write('}')

    def head(self):
        self.out_file.write('digraph Tree {\n')
        self.out_file.write('node [shape=box')
        rounded_filled = []
        if self.filled:
            rounded_filled.append('filled')
        if self.rounded:
            rounded_filled.append('rounded')
        if len(rounded_filled) > 0:
            self.out_file.write(', style="%s", color="black"' % ', '.join(rounded_filled))
        self.out_file.write(', fontname="%s"' % self.fontname)
        self.out_file.write('] ;\n')
        if self.leaves_parallel:
            self.out_file.write('graph [ranksep=equally, splines=polyline] ;\n')
        self.out_file.write('edge [fontname="%s"] ;\n' % self.fontname)
        if self.rotate:
            self.out_file.write('rankdir=LR ;\n')

    def recurse(self, tree, node_id, criterion, parent=None, depth=0):
        if node_id == _tree.TREE_LEAF:
            raise ValueError('Invalid node_id %s' % _tree.TREE_LEAF)
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        if self.max_depth is None or depth <= self.max_depth:
            if left_child == _tree.TREE_LEAF:
                self.ranks['leaves'].append(str(node_id))
            elif str(depth) not in self.ranks:
                self.ranks[str(depth)] = [str(node_id)]
            else:
                self.ranks[str(depth)].append(str(node_id))
            self.out_file.write('%d [label=%s' % (node_id, self.node_to_str(tree, node_id, criterion)))
            if self.filled:
                self.out_file.write(', fillcolor="%s"' % self.get_fill_color(tree, node_id))
            self.out_file.write('] ;\n')
            if parent is not None:
                self.out_file.write('%d -> %d' % (parent, node_id))
                if parent == 0:
                    angles = np.array([45, -45]) * ((self.rotate - 0.5) * -2)
                    self.out_file.write(' [labeldistance=2.5, labelangle=')
                    if node_id == 1:
                        self.out_file.write('%d, headlabel="True"]' % angles[0])
                    else:
                        self.out_file.write('%d, headlabel="False"]' % angles[1])
                self.out_file.write(' ;\n')
            if left_child != _tree.TREE_LEAF:
                self.recurse(tree, left_child, criterion=criterion, parent=node_id, depth=depth + 1)
                self.recurse(tree, right_child, criterion=criterion, parent=node_id, depth=depth + 1)
        else:
            self.ranks['leaves'].append(str(node_id))
            self.out_file.write('%d [label="(...)"' % node_id)
            if self.filled:
                self.out_file.write(', fillcolor="#C0C0C0"')
            self.out_file.write('] ;\n' % node_id)
            if parent is not None:
                self.out_file.write('%d -> %d ;\n' % (parent, node_id))