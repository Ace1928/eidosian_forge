import deap
from copy import copy
def _starting_imports(operators, operators_used, pipeline_module):
    num_op = len(operators_used)
    num_op_root = 0
    for op in operators_used:
        if op != 'CombineDFs':
            tpot_op = get_by_name(op, operators)
            if tpot_op.root:
                num_op_root += 1
        else:
            num_op_root += 1
    if num_op_root > 1:
        if pipeline_module == 'sklearn':
            return {'sklearn.model_selection': ['train_test_split'], 'sklearn.pipeline': ['make_union', 'make_pipeline'], 'tpot.builtins': ['StackingEstimator']}
        else:
            {'sklearn.model_selection': ['train_test_split'], 'sklearn.pipeline': ['make_union'], 'imblearn.pipeline': ['make_pipeline'], 'tpot.builtins': ['StackingEstimator']}
    elif num_op > 1:
        return {'sklearn.model_selection': ['train_test_split'], f'{pipeline_module}.pipeline': ['make_pipeline']}
    else:
        return {'sklearn.model_selection': ['train_test_split']}