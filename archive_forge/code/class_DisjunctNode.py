class DisjunctNode(IntersectNode):
    """
    Create a disjunct node. In order for this node to be true, all of its
    children must evaluate to false
    """

    def to_string(self, with_parens=None):
        with_parens = self._should_use_paren(with_parens)
        ret = super().to_string(with_parens=False)
        if with_parens:
            return '(-' + ret + ')'
        else:
            return '-' + ret