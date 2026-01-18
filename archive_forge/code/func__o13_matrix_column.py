from ..matrix import matrix
def _o13_matrix_column(A, m):
    fAmj = A * m * _adjoint(A)
    return [(fAmj[0][0].real() + fAmj[1][1].real()) / 2, (fAmj[0][0].real() - fAmj[1][1].real()) / 2, fAmj[0][1].real(), fAmj[0][1].imag()]