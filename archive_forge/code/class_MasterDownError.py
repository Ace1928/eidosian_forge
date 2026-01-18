class MasterDownError(ClusterDownError):
    """
    Error indicated MASTERDOWN error received from cluster.
    Link with MASTER is down and replica-serve-stale-data is set to 'no'.
    """
    pass