import warnings
import networkx as nx
def _bidirectional_pred_succ(G, source, target):
    """Bidirectional shortest path helper.

    Returns (pred, succ, w) where
    pred is a dictionary of predecessors from w to the source, and
    succ is a dictionary of successors from w to the target.
    """
    if target == source:
        return ({target: None}, {source: None}, source)
    if G.is_directed():
        Gpred = G.pred
        Gsucc = G.succ
    else:
        Gpred = G.adj
        Gsucc = G.adj
    pred = {source: None}
    succ = {target: None}
    forward_fringe = [source]
    reverse_fringe = [target]
    while forward_fringe and reverse_fringe:
        if len(forward_fringe) <= len(reverse_fringe):
            this_level = forward_fringe
            forward_fringe = []
            for v in this_level:
                for w in Gsucc[v]:
                    if w not in pred:
                        forward_fringe.append(w)
                        pred[w] = v
                    if w in succ:
                        return (pred, succ, w)
        else:
            this_level = reverse_fringe
            reverse_fringe = []
            for v in this_level:
                for w in Gpred[v]:
                    if w not in succ:
                        succ[w] = v
                        reverse_fringe.append(w)
                    if w in pred:
                        return (pred, succ, w)
    raise nx.NetworkXNoPath(f'No path between {source} and {target}.')