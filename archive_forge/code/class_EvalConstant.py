from pyparsing import Word, nums, alphas, Combine, oneOf, \
class EvalConstant(object):
    """Class to evaluate a parsed constant or variable"""
    vars_ = {}

    def __init__(self, tokens):
        self.value = tokens[0]

    def eval(self):
        if self.value in EvalConstant.vars_:
            return EvalConstant.vars_[self.value]
        else:
            return float(self.value)