import threading
from tensorboard import errors
def GraphOpInfo(self, run, graph_id, op_name):
    """Get the information regarding a graph op's creation.

        Args:
          run: Name of the run.
          graph_id: Debugger-generated ID of the graph that contains
            the op in question. This ID is available from other methods
            of this class, e.g., the return value of `GraphExecutionDigests()`.
          op_name: Name of the op.

        Returns:
          A JSON-serializable object containing the information regarding
            the op's creation and its immediate inputs and consumers.

        Raises:
          NotFoundError if the graph_id or op_name does not exist.
        """
    runs = self.Runs()
    if run not in runs:
        return None
    try:
        graph = self._reader.graph_by_id(graph_id)
    except KeyError:
        raise errors.NotFoundError('There is no graph with ID "%s"' % graph_id)
    try:
        op_creation_digest = graph.get_op_creation_digest(op_name)
    except KeyError:
        raise errors.NotFoundError('There is no op named "%s" in graph with ID "%s"' % (op_name, graph_id))
    data_object = self._opCreationDigestToDataObject(op_creation_digest, graph)
    for input_spec in data_object['inputs']:
        try:
            input_op_digest = graph.get_op_creation_digest(input_spec['op_name'])
        except KeyError:
            input_op_digest = None
        if input_op_digest:
            input_spec['data'] = self._opCreationDigestToDataObject(input_op_digest, graph)
    for slot_consumer_specs in data_object['consumers']:
        for consumer_spec in slot_consumer_specs:
            try:
                digest = graph.get_op_creation_digest(consumer_spec['op_name'])
            except KeyError:
                digest = None
            if digest:
                consumer_spec['data'] = self._opCreationDigestToDataObject(digest, graph)
    return data_object