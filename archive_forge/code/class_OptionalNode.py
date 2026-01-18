class OptionalNode(IntersectNode):
    """
    Create an optional node. If this nodes evaluates to true, then the document
    will be rated higher in score/rank.
    """

    def to_string(self, with_parens=None):
        with_parens = self._should_use_paren(with_parens)
        ret = super().to_string(with_parens=False)
        if with_parens:
            return '(~' + ret + ')'
        else:
            return '~' + ret