import pytest
import textwrap
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import numpy as np
from numpy.f2py.tests import util
class TestCharacterString(util.F2PyTest):
    suffix = '.f90'
    fprefix = 'test_character_string'
    length_list = ['1', '3', 'star']
    code = ''
    for length in length_list:
        fsuffix = length
        clength = dict(star='(*)').get(length, length)
        code += textwrap.dedent(f'\n\n        subroutine {fprefix}_input_{fsuffix}(c, o, n)\n          character*{clength}, intent(in) :: c\n          integer n\n          !f2py integer, depend(c), intent(hide) :: n = slen(c)\n          integer*1, dimension(n) :: o\n          !f2py intent(out) o\n          o = transfer(c, o)\n        end subroutine {fprefix}_input_{fsuffix}\n\n        subroutine {fprefix}_output_{fsuffix}(c, o, n)\n          character*{clength}, intent(out) :: c\n          integer n\n          integer*1, dimension(n), intent(in) :: o\n          !f2py integer, depend(o), intent(hide) :: n = len(o)\n          c = transfer(o, c)\n        end subroutine {fprefix}_output_{fsuffix}\n\n        subroutine {fprefix}_array_input_{fsuffix}(c, o, m, n)\n          integer m, i, n\n          character*{clength}, intent(in), dimension(m) :: c\n          !f2py integer, depend(c), intent(hide) :: m = len(c)\n          !f2py integer, depend(c), intent(hide) :: n = f2py_itemsize(c)\n          integer*1, dimension(m, n), intent(out) :: o\n          do i=1,m\n            o(i, :) = transfer(c(i), o(i, :))\n          end do\n        end subroutine {fprefix}_array_input_{fsuffix}\n\n        subroutine {fprefix}_array_output_{fsuffix}(c, o, m, n)\n          character*{clength}, intent(out), dimension(m) :: c\n          integer n\n          integer*1, dimension(m, n), intent(in) :: o\n          !f2py character(f2py_len=n) :: c\n          !f2py integer, depend(o), intent(hide) :: m = len(o)\n          !f2py integer, depend(o), intent(hide) :: n = shape(o, 1)\n          do i=1,m\n            c(i) = transfer(o(i, :), c(i))\n          end do\n        end subroutine {fprefix}_array_output_{fsuffix}\n\n        subroutine {fprefix}_2d_array_input_{fsuffix}(c, o, m1, m2, n)\n          integer m1, m2, i, j, n\n          character*{clength}, intent(in), dimension(m1, m2) :: c\n          !f2py integer, depend(c), intent(hide) :: m1 = len(c)\n          !f2py integer, depend(c), intent(hide) :: m2 = shape(c, 1)\n          !f2py integer, depend(c), intent(hide) :: n = f2py_itemsize(c)\n          integer*1, dimension(m1, m2, n), intent(out) :: o\n          do i=1,m1\n            do j=1,m2\n              o(i, j, :) = transfer(c(i, j), o(i, j, :))\n            end do\n          end do\n        end subroutine {fprefix}_2d_array_input_{fsuffix}\n        ')

    @pytest.mark.parametrize('length', length_list)
    def test_input(self, length):
        fsuffix = {'(*)': 'star'}.get(length, length)
        f = getattr(self.module, self.fprefix + '_input_' + fsuffix)
        a = {'1': 'a', '3': 'abc', 'star': 'abcde' * 3}[length]
        assert_array_equal(f(a), np.array(list(map(ord, a)), dtype='u1'))

    @pytest.mark.parametrize('length', length_list[:-1])
    def test_output(self, length):
        fsuffix = length
        f = getattr(self.module, self.fprefix + '_output_' + fsuffix)
        a = {'1': 'a', '3': 'abc'}[length]
        assert_array_equal(f(np.array(list(map(ord, a)), dtype='u1')), a.encode())

    @pytest.mark.parametrize('length', length_list)
    def test_array_input(self, length):
        fsuffix = length
        f = getattr(self.module, self.fprefix + '_array_input_' + fsuffix)
        a = np.array([{'1': 'a', '3': 'abc', 'star': 'abcde' * 3}[length], {'1': 'A', '3': 'ABC', 'star': 'ABCDE' * 3}[length]], dtype='S')
        expected = np.array([[c for c in s] for s in a], dtype='u1')
        assert_array_equal(f(a), expected)

    @pytest.mark.parametrize('length', length_list)
    def test_array_output(self, length):
        fsuffix = length
        f = getattr(self.module, self.fprefix + '_array_output_' + fsuffix)
        expected = np.array([{'1': 'a', '3': 'abc', 'star': 'abcde' * 3}[length], {'1': 'A', '3': 'ABC', 'star': 'ABCDE' * 3}[length]], dtype='S')
        a = np.array([[c for c in s] for s in expected], dtype='u1')
        assert_array_equal(f(a), expected)

    @pytest.mark.parametrize('length', length_list)
    def test_2d_array_input(self, length):
        fsuffix = length
        f = getattr(self.module, self.fprefix + '_2d_array_input_' + fsuffix)
        a = np.array([[{'1': 'a', '3': 'abc', 'star': 'abcde' * 3}[length], {'1': 'A', '3': 'ABC', 'star': 'ABCDE' * 3}[length]], [{'1': 'f', '3': 'fgh', 'star': 'fghij' * 3}[length], {'1': 'F', '3': 'FGH', 'star': 'FGHIJ' * 3}[length]]], dtype='S')
        expected = np.array([[[c for c in item] for item in row] for row in a], dtype='u1', order='F')
        assert_array_equal(f(a), expected)