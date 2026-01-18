from . import matrix
def _penalty_of_path(path, penalties):

    def penalty(edge):
        if isinstance(edge, ShortEdge):
            return penalties[0]
        if isinstance(edge, MiddleEdge):
            return penalties[1]
        return penalties[2]
    return sum([penalty(edge) for edge in path])