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
class SymmetricParams:

    def __init__(self):
        self.eigs = eigsh
        self.which = ['LM', 'SM', 'LA', 'SA', 'BE']
        self.mattypes = [csr_matrix, aslinearoperator, np.asarray]
        self.sigmas_modes = {None: ['normal'], 0.5: ['normal', 'buckling', 'cayley']}
        N = 6
        np.random.seed(2300)
        Ar = generate_matrix(N, hermitian=True, pos_definite=True).astype('f').astype('d')
        M = generate_matrix(N, hermitian=True, pos_definite=True).astype('f').astype('d')
        Ac = generate_matrix(N, hermitian=True, pos_definite=True, complex_=True).astype('F').astype('D')
        Mc = generate_matrix(N, hermitian=True, pos_definite=True, complex_=True).astype('F').astype('D')
        v0 = np.random.random(N)
        SS = DictWithRepr('std-symmetric')
        SS['mat'] = Ar
        SS['v0'] = v0
        SS['eval'] = eigh(SS['mat'], eigvals_only=True)
        GS = DictWithRepr('gen-symmetric')
        GS['mat'] = Ar
        GS['bmat'] = M
        GS['v0'] = v0
        GS['eval'] = eigh(GS['mat'], GS['bmat'], eigvals_only=True)
        SH = DictWithRepr('std-hermitian')
        SH['mat'] = Ac
        SH['v0'] = v0
        SH['eval'] = eigh(SH['mat'], eigvals_only=True)
        GH = DictWithRepr('gen-hermitian')
        GH['mat'] = Ac
        GH['bmat'] = M
        GH['v0'] = v0
        GH['eval'] = eigh(GH['mat'], GH['bmat'], eigvals_only=True)
        GHc = DictWithRepr('gen-hermitian-Mc')
        GHc['mat'] = Ac
        GHc['bmat'] = Mc
        GHc['v0'] = v0
        GHc['eval'] = eigh(GHc['mat'], GHc['bmat'], eigvals_only=True)
        self.real_test_cases = [SS, GS]
        self.complex_test_cases = [SH, GH, GHc]