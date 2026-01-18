import logging
from functools import partial
from collections import defaultdict
from os.path import splitext
from .diagrams_base import BaseGraph
class NestedGraph(Graph):
    """ Graph creation support for transitions.extensions.nested.HierarchicalGraphMachine. """

    def __init__(self, *args, **kwargs):
        self._cluster_states = []
        super(NestedGraph, self).__init__(*args, **kwargs)

    def set_node_style(self, state, style):
        for state_name in self._get_state_names(state):
            super(NestedGraph, self).set_node_style(state_name, style)

    def set_previous_transition(self, src, dst):
        src_name = self._get_global_name(src.split(self.machine.state_cls.separator))
        dst_name = self._get_global_name(dst.split(self.machine.state_cls.separator))
        super(NestedGraph, self).set_previous_transition(src_name, dst_name)

    def _add_nodes(self, states, container):
        self._add_nested_nodes(states, container, prefix='', default_style='default')

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

    def _add_edges(self, transitions, container):
        edges_attr = defaultdict(lambda: defaultdict(dict))
        for transition in transitions:
            src = transition['source']
            try:
                dst = transition['dest']
            except KeyError:
                dst = src
            if edges_attr[src][dst]:
                attr = edges_attr[src][dst]
                attr[attr['label_pos']] = ' | '.join([edges_attr[src][dst][attr['label_pos']], self._transition_label(transition)])
            else:
                edges_attr[src][dst] = self._create_edge_attr(src, dst, transition)
        for custom_src, dests in self.custom_styles['edge'].items():
            for custom_dst, style in dests.items():
                if style and (custom_src not in edges_attr or custom_dst not in edges_attr[custom_src]):
                    edges_attr[custom_src][custom_dst] = self._create_edge_attr(custom_src, custom_dst, {'trigger': '', 'dest': ''})
        for src, dests in edges_attr.items():
            for dst, attr in dests.items():
                del attr['label_pos']
                style = self.custom_styles['edge'][src][dst]
                attr.update(**self.machine.style_attributes['edge'][style])
                container.edge(attr.pop('source'), attr.pop('dest'), **attr)

    def _create_edge_attr(self, src, dst, transition):
        label_pos = 'label'
        attr = {}
        if src in self._cluster_states:
            attr['ltail'] = 'cluster_' + src
            src_name = src + '_anchor'
            label_pos = 'headlabel'
        else:
            src_name = src
        if dst in self._cluster_states:
            if not src.startswith(dst):
                attr['lhead'] = 'cluster_' + dst
                label_pos = 'taillabel' if label_pos.startswith('l') else 'label'
            dst_name = dst + '_anchor'
        else:
            dst_name = dst
        if 'ltail' in attr and dst_name.startswith(attr['ltail'][8:]):
            del attr['ltail']
        attr[label_pos] = self._transition_label(transition)
        attr['label_pos'] = label_pos
        attr['source'] = src_name
        attr['dest'] = dst_name
        return attr