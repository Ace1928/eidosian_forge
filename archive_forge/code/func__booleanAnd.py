from io import StringIO
def _booleanAnd(self, elem):
    """
        Calculate boolean and of the given expressions given an element.

        @param elem: The element to calculate the value of the expression from.
        """
    return self.lhs.value(elem) and self.rhs.value(elem)