from pyparsing import Word, nums, alphas, Combine, oneOf, \
class EvalAddOp(object):
    """Class to evaluate addition and subtraction expressions"""

    def __init__(self, tokens):
        self.value = tokens[0]

    def eval(self):
        sum = self.value[0].eval()
        for op, val in operatorOperands(self.value[1:]):
            if op == '+':
                sum += val.eval()
            if op == '-':
                sum -= val.eval()
        return sum