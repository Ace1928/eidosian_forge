import numpy as np
import scipy.linalg
def _diff_pade9(A, E, ident):
    b = (17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0, 2162160.0, 110880.0, 3960.0, 90.0, 1.0)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    A4 = np.dot(A2, A2)
    M4 = np.dot(A2, M2) + np.dot(M2, A2)
    A6 = np.dot(A2, A4)
    M6 = np.dot(A4, M2) + np.dot(M4, A2)
    A8 = np.dot(A4, A4)
    M8 = np.dot(A4, M4) + np.dot(M4, A4)
    U = A.dot(b[9] * A8 + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = b[8] * A8 + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
    Lu = A.dot(b[9] * M8 + b[7] * M6 + b[5] * M4 + b[3] * M2) + E.dot(b[9] * A8 + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
    Lv = b[8] * M8 + b[6] * M6 + b[4] * M4 + b[2] * M2
    return (U, V, Lu, Lv)