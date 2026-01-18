def _BreadthFirstSplit(cluster, n):
    """  *Internal Use Only*

  """
    if len(cluster) < n:
        raise ValueError('Cannot split cluster of length %d into %d pieces' % (len(cluster), n))
    if len(cluster) == n:
        return cluster.GetPoints()
    clusters = [cluster]
    nxtIdx = 0
    for _ in range(n - 1):
        while nxtIdx < len(clusters) and len(clusters[nxtIdx]) == 1:
            nxtIdx += 1
        assert nxtIdx < len(clusters)
        children = clusters[nxtIdx].GetChildren()
        children.sort(key=lambda x: x.GetMetric(), reverse=True)
        for child in children:
            clusters.append(child)
        del clusters[nxtIdx]
    return clusters