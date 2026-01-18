import logging
from functools import partial
from collections import defaultdict
from os.path import splitext
from .diagrams_base import BaseGraph
class Graph(BaseGraph):
    """ Graph creation for transitions.core.Machine.
        Attributes:
            custom_styles (dict): A dictionary of styles for the current graph
    """

    def __init__(self, machine):
        self.custom_styles = {}
        self.reset_styling()
        super(Graph, self).__init__(machine)

    def set_previous_transition(self, src, dst):
        self.custom_styles['edge'][src][dst] = 'previous'
        self.set_node_style(src, 'previous')

    def set_node_style(self, state, style):
        self.custom_styles['node'][state.name if hasattr(state, 'name') else state] = style

    def reset_styling(self):
        self.custom_styles = {'edge': defaultdict(lambda: defaultdict(str)), 'node': defaultdict(str)}

    def _add_nodes(self, states, container):
        for state in states:
            style = self.custom_styles['node'][state['name']]
            container.node(state['name'], label=self._convert_state_attributes(state), **self.machine.style_attributes['node'][style])

    def _add_edges(self, transitions, container):
        edge_labels = defaultdict(lambda: defaultdict(list))
        for transition in transitions:
            try:
                dst = transition['dest']
            except KeyError:
                dst = transition['source']
            edge_labels[transition['source']][dst].append(self._transition_label(transition))
        for src, dests in edge_labels.items():
            for dst, labels in dests.items():
                style = self.custom_styles['edge'][src][dst]
                container.edge(src, dst, label=' | '.join(labels), **self.machine.style_attributes['edge'][style])

    def generate(self):
        """ Triggers the generation of a graph. With graphviz backend, this does nothing since graph trees need to be
        build from scratch with the configured styles.
        """
        if not pgv:
            raise Exception('AGraph diagram requires graphviz')

    def get_graph(self, title=None, roi_state=None):
        title = title if title else self.machine.title
        fsm_graph = pgv.Digraph(name=title, node_attr=self.machine.style_attributes['node']['default'], edge_attr=self.machine.style_attributes['edge']['default'], graph_attr=self.machine.style_attributes['graph']['default'])
        fsm_graph.graph_attr.update(**self.machine.machine_attributes)
        fsm_graph.graph_attr['label'] = title
        states, transitions = self._get_elements()
        if roi_state:
            transitions = [t for t in transitions if t['source'] == roi_state or self.custom_styles['edge'][t['source']][t['dest']]]
            state_names = [t for trans in transitions for t in [trans['source'], trans.get('dest', trans['source'])]]
            state_names += [k for k, style in self.custom_styles['node'].items() if style]
            states = _filter_states(states, state_names, self.machine.state_cls)
        self._add_nodes(states, fsm_graph)
        self._add_edges(transitions, fsm_graph)
        setattr(fsm_graph, 'draw', partial(self.draw, fsm_graph))
        return fsm_graph

    def draw(self, graph, filename, format=None, prog='dot', args=''):
        """
        Generates and saves an image of the state machine using graphviz. Note that `prog` and `args` are only part
        of the signature to mimic `Agraph.draw` and thus allow to easily switch between graph backends.
        Args:
            filename (str or file descriptor or stream or None): path and name of image output, file descriptor,
            stream object or None
            format (str): Optional format of the output file
            prog (str): ignored
            args (str): ignored
        Returns:
            None or str: Returns a binary string of the graph when the first parameter (`filename`) is set to None.
        """
        graph.engine = prog
        if filename is None:
            if format is None:
                raise ValueError("Parameter 'format' must not be None when filename is no valid file path.")
            return graph.pipe(format)
        try:
            filename, ext = splitext(filename)
            format = format if format is not None else ext[1:]
            graph.render(filename, format=format if format else 'png', cleanup=True)
        except (TypeError, AttributeError):
            if format is None:
                raise ValueError("Parameter 'format' must not be None when filename is no valid file path.")
            filename.write(graph.pipe(format))
        return None