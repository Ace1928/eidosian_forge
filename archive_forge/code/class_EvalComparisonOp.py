from pyparsing import Word, nums, alphas, Combine, oneOf, \
class EvalComparisonOp(object):
    """Class to evaluate comparison expressions"""
    opMap = {'<': lambda a, b: a < b, '<=': lambda a, b: a <= b, '>': lambda a, b: a > b, '>=': lambda a, b: a >= b, '!=': lambda a, b: a != b, '=': lambda a, b: a == b, 'LT': lambda a, b: a < b, 'LE': lambda a, b: a <= b, 'GT': lambda a, b: a > b, 'GE': lambda a, b: a >= b, 'NE': lambda a, b: a != b, 'EQ': lambda a, b: a == b, '<>': lambda a, b: a != b}

    def __init__(self, tokens):
        self.value = tokens[0]

    def eval(self):
        val1 = self.value[0].eval()
        for op, val in operatorOperands(self.value[1:]):
            fn = EvalComparisonOp.opMap[op]
            val2 = val.eval()
            if not fn(val1, val2):
                break
            val1 = val2
        else:
            return True
        return False