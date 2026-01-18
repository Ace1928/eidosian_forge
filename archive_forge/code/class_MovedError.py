class MovedError(AskError):
    """
    Error indicated MOVED error received from cluster.
    A request sent to a node that doesn't serve this key will be replayed with
    a MOVED error that points to the correct node.
    """
    pass