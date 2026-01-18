class AskError(ResponseError):
    """
    Error indicated ASK error received from cluster.
    When a slot is set as MIGRATING, the node will accept all queries that
    pertain to this hash slot, but only if the key in question exists,
    otherwise the query is forwarded using a -ASK redirection to the node that
    is target of the migration.

    src node: MIGRATING to dst node
        get > ASK error
        ask dst node > ASKING command
    dst node: IMPORTING from src node
        asking command only affects next command
        any op will be allowed after asking command
    """

    def __init__(self, resp):
        """should only redirect to master node"""
        self.args = (resp,)
        self.message = resp
        slot_id, new_node = resp.split(' ')
        host, port = new_node.rsplit(':', 1)
        self.slot_id = int(slot_id)
        self.node_addr = self.host, self.port = (host, int(port))