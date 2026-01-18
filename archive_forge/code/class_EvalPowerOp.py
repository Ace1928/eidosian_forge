from pyparsing import Word, nums, alphas, Combine, oneOf, \
class EvalPowerOp(object):
    """Class to evaluate multiplication and division expressions"""

    def __init__(self, tokens):
        self.value = tokens[0]

    def eval(self):
        res = self.value[-1].eval()
        for val in self.value[-3::-2]:
            res = val.eval() ** res
        return res