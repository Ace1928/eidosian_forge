from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def _intervals_to_quadrants(intervals, f1, f2, s, t, F):
    """Generate a sequence of extended quadrants from a list of critical points. """
    if not intervals:
        return []
    Q = []
    if not f1:
        (a, b), _, _ = intervals[0]
        if a == b == s:
            if len(intervals) == 1:
                if dup_eval(f2, t, F) > 0:
                    return [OO, A2]
                else:
                    return [OO, A4]
            else:
                (a, _), _, _ = intervals[1]
                if dup_eval(f2, (s + a) / 2, F) > 0:
                    Q.extend([OO, A2])
                    f2_sgn = +1
                else:
                    Q.extend([OO, A4])
                    f2_sgn = -1
                intervals = intervals[1:]
        elif dup_eval(f2, s, F) > 0:
            Q.append(A2)
            f2_sgn = +1
        else:
            Q.append(A4)
            f2_sgn = -1
        for (a, _), indices, _ in intervals:
            Q.append(OO)
            if indices[1] % 2 == 1:
                f2_sgn = -f2_sgn
            if a != t:
                if f2_sgn > 0:
                    Q.append(A2)
                else:
                    Q.append(A4)
        return Q
    if not f2:
        (a, b), _, _ = intervals[0]
        if a == b == s:
            if len(intervals) == 1:
                if dup_eval(f1, t, F) > 0:
                    return [OO, A1]
                else:
                    return [OO, A3]
            else:
                (a, _), _, _ = intervals[1]
                if dup_eval(f1, (s + a) / 2, F) > 0:
                    Q.extend([OO, A1])
                    f1_sgn = +1
                else:
                    Q.extend([OO, A3])
                    f1_sgn = -1
                intervals = intervals[1:]
        elif dup_eval(f1, s, F) > 0:
            Q.append(A1)
            f1_sgn = +1
        else:
            Q.append(A3)
            f1_sgn = -1
        for (a, _), indices, _ in intervals:
            Q.append(OO)
            if indices[0] % 2 == 1:
                f1_sgn = -f1_sgn
            if a != t:
                if f1_sgn > 0:
                    Q.append(A1)
                else:
                    Q.append(A3)
        return Q
    re = dup_eval(f1, s, F)
    im = dup_eval(f2, s, F)
    if not re or not im:
        Q.append(_classify_point(re, im))
        if len(intervals) == 1:
            re = dup_eval(f1, t, F)
            im = dup_eval(f2, t, F)
        else:
            (a, _), _, _ = intervals[1]
            re = dup_eval(f1, (s + a) / 2, F)
            im = dup_eval(f2, (s + a) / 2, F)
        intervals = intervals[1:]
    if re > 0:
        f1_sgn = +1
    else:
        f1_sgn = -1
    if im > 0:
        f2_sgn = +1
    else:
        f2_sgn = -1
    sgn = {(+1, +1): Q1, (-1, +1): Q2, (-1, -1): Q3, (+1, -1): Q4}
    Q.append(sgn[f1_sgn, f2_sgn])
    for (a, b), indices, _ in intervals:
        if a == b:
            re = dup_eval(f1, a, F)
            im = dup_eval(f2, a, F)
            cls = _classify_point(re, im)
            if cls is not None:
                Q.append(cls)
        if 0 in indices:
            if indices[0] % 2 == 1:
                f1_sgn = -f1_sgn
        if 1 in indices:
            if indices[1] % 2 == 1:
                f2_sgn = -f2_sgn
        if not (a == b and b == t):
            Q.append(sgn[f1_sgn, f2_sgn])
    return Q