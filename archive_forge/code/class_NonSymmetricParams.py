import threading
import itertools
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from pytest import raises as assert_raises
import pytest
from numpy import dot, conj, random
from scipy.linalg import eig, eigh
from scipy.sparse import csc_matrix, csr_matrix, diags, rand
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._eigen.arpack import (eigs, eigsh, arpack,
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
class NonSymmetricParams:

    def __init__(self):
        self.eigs = eigs
        self.which = ['LM', 'LR', 'LI']
        self.mattypes = [csr_matrix, aslinearoperator, np.asarray]
        self.sigmas_OPparts = {None: [None], 0.1: ['r'], 0.1 + 0.1j: ['r', 'i']}
        N = 6
        np.random.seed(2300)
        Ar = generate_matrix(N).astype('f').astype('d')
        M = generate_matrix(N, hermitian=True, pos_definite=True).astype('f').astype('d')
        Ac = generate_matrix(N, complex_=True).astype('F').astype('D')
        v0 = np.random.random(N)
        SNR = DictWithRepr('std-real-nonsym')
        SNR['mat'] = Ar
        SNR['v0'] = v0
        SNR['eval'] = eig(SNR['mat'], left=False, right=False)
        GNR = DictWithRepr('gen-real-nonsym')
        GNR['mat'] = Ar
        GNR['bmat'] = M
        GNR['v0'] = v0
        GNR['eval'] = eig(GNR['mat'], GNR['bmat'], left=False, right=False)
        SNC = DictWithRepr('std-cmplx-nonsym')
        SNC['mat'] = Ac
        SNC['v0'] = v0
        SNC['eval'] = eig(SNC['mat'], left=False, right=False)
        GNC = DictWithRepr('gen-cmplx-nonsym')
        GNC['mat'] = Ac
        GNC['bmat'] = M
        GNC['v0'] = v0
        GNC['eval'] = eig(GNC['mat'], GNC['bmat'], left=False, right=False)
        self.real_test_cases = [SNR, GNR]
        self.complex_test_cases = [SNC, GNC]