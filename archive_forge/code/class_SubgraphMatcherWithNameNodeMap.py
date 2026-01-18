from typing import Dict, List, Tuple
from torch.fx import Graph, GraphModule, Node
from torch.fx._compatibility import compatibility
from .matcher_utils import InternalMatch, SubgraphMatcher
@compatibility(is_backward_compatible=False)
class SubgraphMatcherWithNameNodeMap(SubgraphMatcher):
    """Extends SubgraphMatcher to support querying the matched subgraph nodes through node name,
    this requires pattern to have specific format (returning and additional dictionary at the output,
    that has node name as key, and the node in the pattern graph as value, see Example for more details)

    Difference with SubgraphMatcher is that it takes a `pattern_gm` GraphModule as input during
    initialization since we need to modify the graph (which requires `recompile` the GraphModule)

    Example::
        def pattern(x, weight):
            conv = F.conv2d(x, weight)
            relu = F.relu(conv)
            return relu, {"conv": conv, "relu": relu}

        def target_graph(x, weight):
            conv = F.conv2d(x, weight)
            relu = F.relu(conv)
            relu *= 2
            return relu

        pattern_gm = capture_pre_autograd_graph(pattern, example_inputs)
        target_gm = capture_pre_autograd_graph(target_graph, example_inputs)
        matcher = SubgraphMatcherWithNameNodeMap(pattern_gm)
        matches = matcher.match(target_gm)
        for match in matches:
            match.name_node_map["conv"].meta["annotation"] = ...

    """

    def __init__(self, pattern_gm: GraphModule, match_output: bool=False, match_placeholder: bool=False, remove_overlapping_matches: bool=True, ignore_literals: bool=False) -> None:
        pattern_gm, name_node_map = _split_to_graph_and_name_node_map(pattern_gm)
        self.name_node_map = name_node_map
        super().__init__(pattern_gm.graph, match_output, match_placeholder, remove_overlapping_matches, ignore_literals)

    def match(self, graph: Graph) -> List[InternalMatch]:
        """The returned InternalMatch will have name_node_map populated with a map
        from node name (str) to the target node, e.g.
        {"conv": target_conv_ndoe, "relu": target_relu_node}

        this requires the pattern graph returns an additional
        output of node name to node, e.g. instead of:
        ```
        def pattern(...):
            ...
            return relu
        ```
        we should do:
        ```
        def pattern(...):
            ...
            return relu, {"conv": conv, "relu": relu}
        ``` instead
        """
        internal_matches = super().match(graph)
        for internal_match in internal_matches:
            for k, n in self.name_node_map.items():
                internal_match.name_node_map[k] = internal_match.nodes_map[n]
        return internal_matches