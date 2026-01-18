def FindClusterCentroidFromDists(cluster, dists):
    """ find the point in a cluster which has the smallest summed
     Euclidean distance to all others

   **Arguments**

     - cluster: the cluster to work with

     - dists: the distance matrix to use for the points

   **Returns**

     - the index of the centroid point

  """
    children = cluster.GetPoints()
    pts = [x.GetData() for x in children]
    best = 1e+24
    bestIdx = -1
    for pt in pts:
        dAccum = 0.0
        for other in pts:
            if other != pt:
                if other > pt:
                    row, col = (pt, other)
                else:
                    row, col = (other, pt)
                dAccum += dists[col * (col - 1) / 2 + row]
                if dAccum >= best:
                    break
        if dAccum < best:
            best = dAccum
            bestIdx = pt
    for i in range(len(pts)):
        pt = pts[i]
        if pt != bestIdx:
            if pt > bestIdx:
                row, col = (bestIdx, pt)
            else:
                row, col = (pt, bestIdx)
            children[i]._distToCenter = dists[col * (col - 1) / 2 + row]
        else:
            children[i]._distToCenter = 0.0
        children[i]._clustCenter = bestIdx
    cluster._clustCenter = bestIdx
    cluster._distToCenter = 0.0
    return bestIdx