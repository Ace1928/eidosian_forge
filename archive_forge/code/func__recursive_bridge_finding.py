def _recursive_bridge_finding(G, u, v):
    global cnt
    global low
    global preorder
    global bridges
    cnt += 1
    preorder[v] = cnt
    low[v] = preorder[v]
    for w in G.neighbor_iterator(v):
        if preorder[w] == -1:
            _recursive_bridge_finding(G, v, w)
            low[v] = min(low[v], low[w])
            if low[w] == preorder[w]:
                bridges.append((min(v, w), max(v, w), None))
        elif u != w or connections(G, u, v) > 1:
            low[v] = min(low[v], preorder[w])