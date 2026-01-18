from io import StringIO
def _booleanOr(self, elem):
    """
        Calculate boolean or of the given expressions given an element.

        @param elem: The element to calculate the value of the expression from.
        """
    return self.lhs.value(elem) or self.rhs.value(elem)