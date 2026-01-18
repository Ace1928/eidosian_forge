from collections.abc import Iterable
from io import StringIO
from numbers import Integral
import numpy as np
from ..base import is_classifier
from ..utils._param_validation import HasMethods, Interval, StrOptions, validate_params
from ..utils.validation import check_array, check_is_fitted
from . import DecisionTreeClassifier, DecisionTreeRegressor, _criterion, _tree
from ._reingold_tilford import Tree, buchheim
class _BaseTreeExporter:

    def __init__(self, max_depth=None, feature_names=None, class_names=None, label='all', filled=False, impurity=True, node_ids=False, proportion=False, rounded=False, precision=3, fontsize=None):
        self.max_depth = max_depth
        self.feature_names = feature_names
        self.class_names = class_names
        self.label = label
        self.filled = filled
        self.impurity = impurity
        self.node_ids = node_ids
        self.proportion = proportion
        self.rounded = rounded
        self.precision = precision
        self.fontsize = fontsize

    def get_color(self, value):
        if self.colors['bounds'] is None:
            color = list(self.colors['rgb'][np.argmax(value)])
            sorted_values = sorted(value, reverse=True)
            if len(sorted_values) == 1:
                alpha = 0.0
            else:
                alpha = (sorted_values[0] - sorted_values[1]) / (1 - sorted_values[1])
        else:
            color = list(self.colors['rgb'][0])
            alpha = (value - self.colors['bounds'][0]) / (self.colors['bounds'][1] - self.colors['bounds'][0])
        color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
        return '#%2x%2x%2x' % tuple(color)

    def get_fill_color(self, tree, node_id):
        if 'rgb' not in self.colors:
            self.colors['rgb'] = _color_brew(tree.n_classes[0])
            if tree.n_outputs != 1:
                self.colors['bounds'] = (np.min(-tree.impurity), np.max(-tree.impurity))
            elif tree.n_classes[0] == 1 and len(np.unique(tree.value)) != 1:
                self.colors['bounds'] = (np.min(tree.value), np.max(tree.value))
        if tree.n_outputs == 1:
            node_val = tree.value[node_id][0, :]
            if tree.n_classes[0] == 1 and isinstance(node_val, Iterable) and (self.colors['bounds'] is not None):
                node_val = node_val.item()
        else:
            node_val = -tree.impurity[node_id]
        return self.get_color(node_val)

    def node_to_str(self, tree, node_id, criterion):
        if tree.n_outputs == 1:
            value = tree.value[node_id][0, :]
        else:
            value = tree.value[node_id]
        labels = self.label == 'root' and node_id == 0 or self.label == 'all'
        characters = self.characters
        node_string = characters[-1]
        if self.node_ids:
            if labels:
                node_string += 'node '
            node_string += characters[0] + str(node_id) + characters[4]
        if tree.children_left[node_id] != _tree.TREE_LEAF:
            if self.feature_names is not None:
                feature = self.feature_names[tree.feature[node_id]]
            else:
                feature = 'x%s%s%s' % (characters[1], tree.feature[node_id], characters[2])
            node_string += '%s %s %s%s' % (feature, characters[3], round(tree.threshold[node_id], self.precision), characters[4])
        if self.impurity:
            if isinstance(criterion, _criterion.FriedmanMSE):
                criterion = 'friedman_mse'
            elif isinstance(criterion, _criterion.MSE) or criterion == 'squared_error':
                criterion = 'squared_error'
            elif not isinstance(criterion, str):
                criterion = 'impurity'
            if labels:
                node_string += '%s = ' % criterion
            node_string += str(round(tree.impurity[node_id], self.precision)) + characters[4]
        if labels:
            node_string += 'samples = '
        if self.proportion:
            percent = 100.0 * tree.n_node_samples[node_id] / float(tree.n_node_samples[0])
            node_string += str(round(percent, 1)) + '%' + characters[4]
        else:
            node_string += str(tree.n_node_samples[node_id]) + characters[4]
        if not self.proportion and tree.n_classes[0] != 1:
            value = value * tree.weighted_n_node_samples[node_id]
        if labels:
            node_string += 'value = '
        if tree.n_classes[0] == 1:
            value_text = np.around(value, self.precision)
        elif self.proportion:
            value_text = np.around(value, self.precision)
        elif np.all(np.equal(np.mod(value, 1), 0)):
            value_text = value.astype(int)
        else:
            value_text = np.around(value, self.precision)
        value_text = str(value_text.astype('S32')).replace("b'", "'")
        value_text = value_text.replace("' '", ', ').replace("'", '')
        if tree.n_classes[0] == 1 and tree.n_outputs == 1:
            value_text = value_text.replace('[', '').replace(']', '')
        value_text = value_text.replace('\n ', characters[4])
        node_string += value_text + characters[4]
        if self.class_names is not None and tree.n_classes[0] != 1 and (tree.n_outputs == 1):
            if labels:
                node_string += 'class = '
            if self.class_names is not True:
                class_name = self.class_names[np.argmax(value)]
            else:
                class_name = 'y%s%s%s' % (characters[1], np.argmax(value), characters[2])
            node_string += class_name
        if node_string.endswith(characters[4]):
            node_string = node_string[:-len(characters[4])]
        return node_string + characters[5]