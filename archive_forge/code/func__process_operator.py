import deap
from copy import copy
def _process_operator(operator, operators, depth=0):
    steps = []
    op_name = operator[0]
    if op_name == 'CombineDFs':
        steps.append(_combine_dfs(operator[1], operator[2], operators))
    else:
        input_name, args = (operator[1], operator[2:])
        tpot_op = get_by_name(op_name, operators)
        if input_name != 'input_matrix':
            steps.extend(_process_operator(input_name, operators, depth + 1))
        if tpot_op.root and depth > 0:
            steps.append('StackingEstimator(estimator={})'.format(tpot_op.export(*args)))
        else:
            steps.append(tpot_op.export(*args))
    return steps