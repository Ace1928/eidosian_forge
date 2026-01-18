import cupy
def _endprint(x, flag, fval, maxfun, xtol, disp):
    if flag == 0:
        if disp > 1:
            print('\nOptimization terminated successfully;\nThe returned value satisfies the termination criteria\n(using xtol = ', xtol, ')')
    if flag == 1:
        if disp:
            print('\nMaximum number of function evaluations exceeded --- increase maxfun argument.\n')
    if flag == 2:
        if disp:
            print('\n{}'.format(_status_message['nan']))
    return