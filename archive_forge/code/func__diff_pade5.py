import numpy as np
import scipy.linalg
def _diff_pade5(A, E, ident):
    b = (30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    A4 = np.dot(A2, A2)
    M4 = np.dot(A2, M2) + np.dot(M2, A2)
    U = A.dot(b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = b[4] * A4 + b[2] * A2 + b[0] * ident
    Lu = A.dot(b[5] * M4 + b[3] * M2) + E.dot(b[5] * A4 + b[3] * A2 + b[1] * ident)
    Lv = b[4] * M4 + b[2] * M2
    return (U, V, Lu, Lv)