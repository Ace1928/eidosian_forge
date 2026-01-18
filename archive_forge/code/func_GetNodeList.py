def GetNodeList(cluster):
    """returns an ordered list of all nodes below cluster

  the ordering is done using the lengths of the child nodes

   **Arguments**

     - cluster: the cluster in question

   **Returns**

     - a list of the leaves below this cluster

  """
    if len(cluster) == 1:
        return [cluster]
    else:
        children = cluster.GetChildren()
        children.sort(key=lambda x: len(x), reverse=True)
        res = []
        for child in children:
            res += GetNodeList(child)
        res += [cluster]
        return res