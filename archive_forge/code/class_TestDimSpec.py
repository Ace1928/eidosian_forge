import importlib
import codecs
import time
import unicodedata
import pytest
import numpy as np
from numpy.f2py.crackfortran import markinnerspaces, nameargspattern
from . import util
from numpy.f2py import crackfortran
import textwrap
import contextlib
import io
class TestDimSpec(util.F2PyTest):
    """This test suite tests various expressions that are used as dimension
    specifications.

    There exists two usage cases where analyzing dimensions
    specifications are important.

    In the first case, the size of output arrays must be defined based
    on the inputs to a Fortran function. Because Fortran supports
    arbitrary bases for indexing, for instance, `arr(lower:upper)`,
    f2py has to evaluate an expression `upper - lower + 1` where
    `lower` and `upper` are arbitrary expressions of input parameters.
    The evaluation is performed in C, so f2py has to translate Fortran
    expressions to valid C expressions (an alternative approach is
    that a developer specifies the corresponding C expressions in a
    .pyf file).

    In the second case, when user provides an input array with a given
    size but some hidden parameters used in dimensions specifications
    need to be determined based on the input array size. This is a
    harder problem because f2py has to solve the inverse problem: find
    a parameter `p` such that `upper(p) - lower(p) + 1` equals to the
    size of input array. In the case when this equation cannot be
    solved (e.g. because the input array size is wrong), raise an
    error before calling the Fortran function (that otherwise would
    likely crash Python process when the size of input arrays is
    wrong). f2py currently supports this case only when the equation
    is linear with respect to unknown parameter.

    """
    suffix = '.f90'
    code_template = textwrap.dedent('\n      function get_arr_size_{count}(a, n) result (length)\n        integer, intent(in) :: n\n        integer, dimension({dimspec}), intent(out) :: a\n        integer length\n        length = size(a)\n      end function\n\n      subroutine get_inv_arr_size_{count}(a, n)\n        integer :: n\n        ! the value of n is computed in f2py wrapper\n        !f2py intent(out) n\n        integer, dimension({dimspec}), intent(in) :: a\n      end subroutine\n    ')
    linear_dimspecs = ['n', '2*n', '2:n', 'n/2', '5 - n/2', '3*n:20', 'n*(n+1):n*(n+5)', '2*n, n']
    nonlinear_dimspecs = ['2*n:3*n*n+2*n']
    all_dimspecs = linear_dimspecs + nonlinear_dimspecs
    code = ''
    for count, dimspec in enumerate(all_dimspecs):
        lst = [d.split(':')[0] if ':' in d else '1' for d in dimspec.split(',')]
        code += code_template.format(count=count, dimspec=dimspec, first=', '.join(lst))

    @pytest.mark.parametrize('dimspec', all_dimspecs)
    def test_array_size(self, dimspec):
        count = self.all_dimspecs.index(dimspec)
        get_arr_size = getattr(self.module, f'get_arr_size_{count}')
        for n in [1, 2, 3, 4, 5]:
            sz, a = get_arr_size(n)
            assert a.size == sz

    @pytest.mark.parametrize('dimspec', all_dimspecs)
    def test_inv_array_size(self, dimspec):
        count = self.all_dimspecs.index(dimspec)
        get_arr_size = getattr(self.module, f'get_arr_size_{count}')
        get_inv_arr_size = getattr(self.module, f'get_inv_arr_size_{count}')
        for n in [1, 2, 3, 4, 5]:
            sz, a = get_arr_size(n)
            if dimspec in self.nonlinear_dimspecs:
                n1 = get_inv_arr_size(a, n)
            else:
                n1 = get_inv_arr_size(a)
            sz1, _ = get_arr_size(n1)
            assert sz == sz1, (n, n1, sz, sz1)