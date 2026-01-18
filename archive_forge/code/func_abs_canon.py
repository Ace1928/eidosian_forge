from cvxpy.expressions.variable import Variable
def abs_canon(expr, args):
    x = args[0]
    t = Variable(expr.shape)
    constraints = [t >= x, t >= -x]
    return (t, constraints)