from collections.abc import Iterable
from io import StringIO
from numbers import Integral
import numpy as np
from ..base import is_classifier
from ..utils._param_validation import HasMethods, Interval, StrOptions, validate_params
from ..utils.validation import check_array, check_is_fitted
from . import DecisionTreeClassifier, DecisionTreeRegressor, _criterion, _tree
from ._reingold_tilford import Tree, buchheim
class _MPLTreeExporter(_BaseTreeExporter):

    def __init__(self, max_depth=None, feature_names=None, class_names=None, label='all', filled=False, impurity=True, node_ids=False, proportion=False, rounded=False, precision=3, fontsize=None):
        super().__init__(max_depth=max_depth, feature_names=feature_names, class_names=class_names, label=label, filled=filled, impurity=impurity, node_ids=node_ids, proportion=proportion, rounded=rounded, precision=precision)
        self.fontsize = fontsize
        self.ranks = {'leaves': []}
        self.colors = {'bounds': None}
        self.characters = ['#', '[', ']', '<=', '\n', '', '']
        self.bbox_args = dict()
        if self.rounded:
            self.bbox_args['boxstyle'] = 'round'
        self.arrow_args = dict(arrowstyle='<-')

    def _make_tree(self, node_id, et, criterion, depth=0):
        name = self.node_to_str(et, node_id, criterion=criterion)
        if et.children_left[node_id] != _tree.TREE_LEAF and (self.max_depth is None or depth <= self.max_depth):
            children = [self._make_tree(et.children_left[node_id], et, criterion, depth=depth + 1), self._make_tree(et.children_right[node_id], et, criterion, depth=depth + 1)]
        else:
            return Tree(name, node_id)
        return Tree(name, node_id, *children)

    def export(self, decision_tree, ax=None):
        import matplotlib.pyplot as plt
        from matplotlib.text import Annotation
        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.set_axis_off()
        my_tree = self._make_tree(0, decision_tree.tree_, decision_tree.criterion)
        draw_tree = buchheim(my_tree)
        max_x, max_y = draw_tree.max_extents() + 1
        ax_width = ax.get_window_extent().width
        ax_height = ax.get_window_extent().height
        scale_x = ax_width / max_x
        scale_y = ax_height / max_y
        self.recurse(draw_tree, decision_tree.tree_, ax, max_x, max_y)
        anns = [ann for ann in ax.get_children() if isinstance(ann, Annotation)]
        renderer = ax.figure.canvas.get_renderer()
        for ann in anns:
            ann.update_bbox_position_size(renderer)
        if self.fontsize is None:
            extents = [ann.get_bbox_patch().get_window_extent() for ann in anns]
            max_width = max([extent.width for extent in extents])
            max_height = max([extent.height for extent in extents])
            size = anns[0].get_fontsize() * min(scale_x / max_width, scale_y / max_height)
            for ann in anns:
                ann.set_fontsize(size)
        return anns

    def recurse(self, node, tree, ax, max_x, max_y, depth=0):
        import matplotlib.pyplot as plt
        kwargs = dict(bbox=self.bbox_args.copy(), ha='center', va='center', zorder=100 - 10 * depth, xycoords='axes fraction', arrowprops=self.arrow_args.copy())
        kwargs['arrowprops']['edgecolor'] = plt.rcParams['text.color']
        if self.fontsize is not None:
            kwargs['fontsize'] = self.fontsize
        xy = ((node.x + 0.5) / max_x, (max_y - node.y - 0.5) / max_y)
        if self.max_depth is None or depth <= self.max_depth:
            if self.filled:
                kwargs['bbox']['fc'] = self.get_fill_color(tree, node.tree.node_id)
            else:
                kwargs['bbox']['fc'] = ax.get_facecolor()
            if node.parent is None:
                ax.annotate(node.tree.label, xy, **kwargs)
            else:
                xy_parent = ((node.parent.x + 0.5) / max_x, (max_y - node.parent.y - 0.5) / max_y)
                ax.annotate(node.tree.label, xy_parent, xy, **kwargs)
            for child in node.children:
                self.recurse(child, tree, ax, max_x, max_y, depth=depth + 1)
        else:
            xy_parent = ((node.parent.x + 0.5) / max_x, (max_y - node.parent.y - 0.5) / max_y)
            kwargs['bbox']['fc'] = 'grey'
            ax.annotate('\n  (...)  \n', xy_parent, xy, **kwargs)