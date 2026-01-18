import threading
def _lazily_initialize(self):
    """Initialize the graph and session, if this has not yet been done."""
    import tensorflow.compat.v1 as tf
    with self._initialization_lock:
        if self._session:
            return
        graph = tf.Graph()
        with graph.as_default():
            self.initialize_graph()
        config = tf.ConfigProto(device_count={'GPU': 0})
        self._session = tf.Session(graph=graph, config=config)