import math
from copy import deepcopy
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from .basic import Booster, _data_from_pandas, _is_zero, _log_warning, _MissingType
from .compat import GRAPHVIZ_INSTALLED, MATPLOTLIB_INSTALLED, pd_DataFrame
from .sklearn import LGBMModel
def _to_graphviz(tree_info: Dict[str, Any], show_info: List[str], feature_names: Union[List[str], None], precision: Optional[int], orientation: str, constraints: Optional[List[int]], example_case: Optional[Union[np.ndarray, pd_DataFrame]], max_category_values: int, **kwargs: Any) -> Any:
    """Convert specified tree to graphviz instance.

    See:
      - https://graphviz.readthedocs.io/en/stable/api.html#digraph
    """
    if GRAPHVIZ_INSTALLED:
        from graphviz import Digraph
    else:
        raise ImportError('You must install graphviz and restart your session to plot tree.')

    def add(root: Dict[str, Any], total_count: int, parent: Optional[str], decision: Optional[str], highlight: bool) -> None:
        """Recursively add node or edge."""
        fillcolor = 'white'
        style = ''
        tooltip = None
        if highlight:
            color = 'blue'
            penwidth = '3'
        else:
            color = 'black'
            penwidth = '1'
        if 'split_index' in root:
            shape = 'rectangle'
            l_dec = 'yes'
            r_dec = 'no'
            threshold = root['threshold']
            if root['decision_type'] == '<=':
                operator = '&#8804;'
            elif root['decision_type'] == '==':
                operator = '='
            else:
                raise ValueError('Invalid decision type in tree model.')
            name = f'split{root['split_index']}'
            split_feature = root['split_feature']
            if feature_names is not None:
                label = f'<B>{feature_names[split_feature]}</B> {operator}'
            else:
                label = f'feature <B>{split_feature}</B> {operator} '
            direction = None
            if example_case is not None:
                if root['decision_type'] == '==':
                    direction = _determine_direction_for_categorical_split(fval=example_case[split_feature], thresholds=root['threshold'])
                else:
                    direction = _determine_direction_for_numeric_split(fval=example_case[split_feature], threshold=root['threshold'], missing_type_str=root['missing_type'], default_left=root['default_left'])
            if root['decision_type'] == '==':
                category_values = root['threshold'].split('||')
                if len(category_values) > max_category_values:
                    tooltip = root['threshold']
                    threshold = '||'.join(category_values[:2]) + '||...||' + category_values[-1]
            label += f'<B>{_float2str(threshold, precision)}</B>'
            for info in ['split_gain', 'internal_value', 'internal_weight', 'internal_count', 'data_percentage']:
                if info in show_info:
                    output = info.split('_')[-1]
                    if info in {'split_gain', 'internal_value', 'internal_weight'}:
                        label += f'<br/>{_float2str(root[info], precision)} {output}'
                    elif info == 'internal_count':
                        label += f'<br/>{output}: {root[info]}'
                    elif info == 'data_percentage':
                        label += f'<br/>{_float2str(root['internal_count'] / total_count * 100, 2)}% of data'
            if constraints:
                if constraints[root['split_feature']] == 1:
                    fillcolor = '#ddffdd'
                if constraints[root['split_feature']] == -1:
                    fillcolor = '#ffdddd'
                style = 'filled'
            label = f'<{label}>'
            add(root=root['left_child'], total_count=total_count, parent=name, decision=l_dec, highlight=highlight and direction == 'left')
            add(root=root['right_child'], total_count=total_count, parent=name, decision=r_dec, highlight=highlight and direction == 'right')
        else:
            shape = 'ellipse'
            name = f'leaf{root['leaf_index']}'
            label = f'leaf {root['leaf_index']}: '
            label += f'<B>{_float2str(root['leaf_value'], precision)}</B>'
            if 'leaf_weight' in show_info:
                label += f'<br/>{_float2str(root['leaf_weight'], precision)} weight'
            if 'leaf_count' in show_info:
                label += f'<br/>count: {root['leaf_count']}'
            if 'data_percentage' in show_info:
                label += f'<br/>{_float2str(root['leaf_count'] / total_count * 100, 2)}% of data'
            label = f'<{label}>'
        graph.node(name, label=label, shape=shape, style=style, fillcolor=fillcolor, color=color, penwidth=penwidth, tooltip=tooltip)
        if parent is not None:
            graph.edge(parent, name, decision, color=color, penwidth=penwidth)
    graph = Digraph(**kwargs)
    rankdir = 'LR' if orientation == 'horizontal' else 'TB'
    graph.attr('graph', nodesep='0.05', ranksep='0.3', rankdir=rankdir)
    if 'internal_count' in tree_info['tree_structure']:
        add(root=tree_info['tree_structure'], total_count=tree_info['tree_structure']['internal_count'], parent=None, decision=None, highlight=example_case is not None)
    else:
        raise Exception('Cannot plot trees with no split')
    if constraints:
        legend = '<\n            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">\n             <TR>\n              <TD COLSPAN="2"><B>Monotone constraints</B></TD>\n             </TR>\n             <TR>\n              <TD>Increasing</TD>\n              <TD BGCOLOR="#ddffdd"></TD>\n             </TR>\n             <TR>\n              <TD>Decreasing</TD>\n              <TD BGCOLOR="#ffdddd"></TD>\n             </TR>\n            </TABLE>\n           >'
        graph.node('legend', label=legend, shape='rectangle', color='white')
    return graph