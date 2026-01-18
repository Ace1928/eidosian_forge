class BFSVisitor:
    """A visitor object that is invoked at the event-points inside the
    :func:`~rustworkx.bfs_search` algorithm. By default, it performs no
    action, and should be used as a base class in order to be useful.
    """

    def discover_vertex(self, v):
        """
        This is invoked when a vertex is encountered for the first time.
        """
        return

    def finish_vertex(self, v):
        """
        This is invoked on vertex `v` after all of its out edges have been examined.
        """
        return

    def tree_edge(self, e):
        """
        This is invoked on each edge as it becomes a member of the edges
        that form the search tree.
        """
        return

    def non_tree_edge(self, e):
        """
        This is invoked on back or cross edges for directed graphs and cross edges
        for undirected graphs.
        """
        return

    def gray_target_edge(self, e):
        """
        This is invoked on the subset of non-tree edges whose target vertex is
        colored gray at the time of examination.
        The color gray indicates that the vertex is currently in the queue.
        """
        return

    def black_target_edge(self, e):
        """
        This is invoked on the subset of non-tree edges whose target vertex is
        colored black at the time of examination.
        The color black indicates that the vertex has been removed from the queue.
        """
        return