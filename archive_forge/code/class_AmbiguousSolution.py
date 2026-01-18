class AmbiguousSolution(NetworkXException):
    """Raised if more than one valid solution exists for an intermediary step
    of an algorithm.

    In the face of ambiguity, refuse the temptation to guess.
    This may occur, for example, when trying to determine the
    bipartite node sets in a disconnected bipartite graph when
    computing bipartite matchings.

    """