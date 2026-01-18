import os
import numpy
from warnings import warn
from scipy.odr import __odrpack
def _report_error(info):
    """ Interprets the return code of the odr routine.

    Parameters
    ----------
    info : int
        The return code of the odr routine.

    Returns
    -------
    problems : list(str)
        A list of messages about why the odr() routine stopped.
    """
    stopreason = ('Blank', 'Sum of squares convergence', 'Parameter convergence', 'Both sum of squares and parameter convergence', 'Iteration limit reached')[info % 5]
    if info >= 5:
        I = (info // 10000 % 10, info // 1000 % 10, info // 100 % 10, info // 10 % 10, info % 10)
        problems = []
        if I[0] == 0:
            if I[1] != 0:
                problems.append('Derivatives possibly not correct')
            if I[2] != 0:
                problems.append('Error occurred in callback')
            if I[3] != 0:
                problems.append('Problem is not full rank at solution')
            problems.append(stopreason)
        elif I[0] == 1:
            if I[1] != 0:
                problems.append('N < 1')
            if I[2] != 0:
                problems.append('M < 1')
            if I[3] != 0:
                problems.append('NP < 1 or NP > N')
            if I[4] != 0:
                problems.append('NQ < 1')
        elif I[0] == 2:
            if I[1] != 0:
                problems.append('LDY and/or LDX incorrect')
            if I[2] != 0:
                problems.append('LDWE, LD2WE, LDWD, and/or LD2WD incorrect')
            if I[3] != 0:
                problems.append('LDIFX, LDSTPD, and/or LDSCLD incorrect')
            if I[4] != 0:
                problems.append('LWORK and/or LIWORK too small')
        elif I[0] == 3:
            if I[1] != 0:
                problems.append('STPB and/or STPD incorrect')
            if I[2] != 0:
                problems.append('SCLB and/or SCLD incorrect')
            if I[3] != 0:
                problems.append('WE incorrect')
            if I[4] != 0:
                problems.append('WD incorrect')
        elif I[0] == 4:
            problems.append('Error in derivatives')
        elif I[0] == 5:
            problems.append('Error occurred in callback')
        elif I[0] == 6:
            problems.append('Numerical error detected')
        return problems
    else:
        return [stopreason]