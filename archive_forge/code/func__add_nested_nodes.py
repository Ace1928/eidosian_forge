import logging
from functools import partial
from collections import defaultdict
from os.path import splitext
from .diagrams_base import BaseGraph
def _add_nested_nodes(self, states, container, prefix, default_style):
    for state in states:
        name = prefix + state['name']
        label = self._convert_state_attributes(state)
        if state.get('children', []):
            cluster_name = 'cluster_' + name
            attr = {'label': label, 'rank': 'source'}
            attr.update(**self.machine.style_attributes['graph'][self.custom_styles['node'][name] or default_style])
            with container.subgraph(name=cluster_name, graph_attr=attr) as sub:
                self._cluster_states.append(name)
                is_parallel = isinstance(state.get('initial', ''), list)
                with sub.subgraph(name=cluster_name + '_root', graph_attr={'label': '', 'color': 'None', 'rank': 'min'}) as root:
                    root.node(name + '_anchor', shape='point', fillcolor='black', width='0.0' if is_parallel else '0.1')
                self._add_nested_nodes(state['children'], sub, default_style='parallel' if is_parallel else 'default', prefix=prefix + state['name'] + self.machine.state_cls.separator)
        else:
            style = self.machine.style_attributes['node'][default_style].copy()
            style.update(self.machine.style_attributes['node'][self.custom_styles['node'][name] or default_style])
            container.node(name, label=label, **style)