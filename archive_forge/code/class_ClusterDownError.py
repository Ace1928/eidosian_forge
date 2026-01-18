class ClusterDownError(ClusterError, ResponseError):
    """
    Error indicated CLUSTERDOWN error received from cluster.
    By default Redis Cluster nodes stop accepting queries if they detect there
    is at least a hash slot uncovered (no available node is serving it).
    This way if the cluster is partially down (for example a range of hash
    slots are no longer covered) the entire cluster eventually becomes
    unavailable. It automatically returns available as soon as all the slots
    are covered again.
    """

    def __init__(self, resp):
        self.args = (resp,)
        self.message = resp