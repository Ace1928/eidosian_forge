from io import StringIO
class CompareValue:

    def __init__(self, lhs, op, rhs):
        self.lhs = lhs
        self.rhs = rhs
        if op == '=':
            self.value = self._compareEqual
        else:
            self.value = self._compareNotEqual

    def _compareEqual(self, elem):
        return self.lhs.value(elem) == self.rhs.value(elem)

    def _compareNotEqual(self, elem):
        return self.lhs.value(elem) != self.rhs.value(elem)