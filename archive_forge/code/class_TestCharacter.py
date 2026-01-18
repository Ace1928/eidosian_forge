import pytest
import textwrap
from numpy.testing import assert_array_equal, assert_equal, assert_raises
import numpy as np
from numpy.f2py.tests import util
class TestCharacter(util.F2PyTest):
    suffix = '.f90'
    fprefix = 'test_character'
    code = textwrap.dedent(f'\n       subroutine {fprefix}_input(c, o)\n          character, intent(in) :: c\n          integer*1 o\n          !f2py intent(out) o\n          o = transfer(c, o)\n       end subroutine {fprefix}_input\n\n       subroutine {fprefix}_output(c, o)\n          character :: c\n          integer*1, intent(in) :: o\n          !f2py intent(out) c\n          c = transfer(o, c)\n       end subroutine {fprefix}_output\n\n       subroutine {fprefix}_input_output(c, o)\n          character, intent(in) :: c\n          character o\n          !f2py intent(out) o\n          o = c\n       end subroutine {fprefix}_input_output\n\n       subroutine {fprefix}_inout(c, n)\n          character :: c, n\n          !f2py intent(in) n\n          !f2py intent(inout) c\n          c = n\n       end subroutine {fprefix}_inout\n\n       function {fprefix}_return(o) result (c)\n          character :: c\n          character, intent(in) :: o\n          c = transfer(o, c)\n       end function {fprefix}_return\n\n       subroutine {fprefix}_array_input(c, o)\n          character, intent(in) :: c(3)\n          integer*1 o(3)\n          !f2py intent(out) o\n          integer i\n          do i=1,3\n            o(i) = transfer(c(i), o(i))\n          end do\n       end subroutine {fprefix}_array_input\n\n       subroutine {fprefix}_2d_array_input(c, o)\n          character, intent(in) :: c(2, 3)\n          integer*1 o(2, 3)\n          !f2py intent(out) o\n          integer i, j\n          do i=1,2\n            do j=1,3\n              o(i, j) = transfer(c(i, j), o(i, j))\n            end do\n          end do\n       end subroutine {fprefix}_2d_array_input\n\n       subroutine {fprefix}_array_output(c, o)\n          character :: c(3)\n          integer*1, intent(in) :: o(3)\n          !f2py intent(out) c\n          do i=1,3\n            c(i) = transfer(o(i), c(i))\n          end do\n       end subroutine {fprefix}_array_output\n\n       subroutine {fprefix}_array_inout(c, n)\n          character :: c(3), n(3)\n          !f2py intent(in) n(3)\n          !f2py intent(inout) c(3)\n          do i=1,3\n            c(i) = n(i)\n          end do\n       end subroutine {fprefix}_array_inout\n\n       subroutine {fprefix}_2d_array_inout(c, n)\n          character :: c(2, 3), n(2, 3)\n          !f2py intent(in) n(2, 3)\n          !f2py intent(inout) c(2. 3)\n          integer i, j\n          do i=1,2\n            do j=1,3\n              c(i, j) = n(i, j)\n            end do\n          end do\n       end subroutine {fprefix}_2d_array_inout\n\n       function {fprefix}_array_return(o) result (c)\n          character, dimension(3) :: c\n          character, intent(in) :: o(3)\n          do i=1,3\n            c(i) = o(i)\n          end do\n       end function {fprefix}_array_return\n\n       function {fprefix}_optional(o) result (c)\n          character, intent(in) :: o\n          !f2py character o = "a"\n          character :: c\n          c = o\n       end function {fprefix}_optional\n    ')

    @pytest.mark.parametrize('dtype', ['c', 'S1'])
    def test_input(self, dtype):
        f = getattr(self.module, self.fprefix + '_input')
        assert_equal(f(np.array('a', dtype=dtype)), ord('a'))
        assert_equal(f(np.array(b'a', dtype=dtype)), ord('a'))
        assert_equal(f(np.array(['a'], dtype=dtype)), ord('a'))
        assert_equal(f(np.array('abc', dtype=dtype)), ord('a'))
        assert_equal(f(np.array([['a']], dtype=dtype)), ord('a'))

    def test_input_varia(self):
        f = getattr(self.module, self.fprefix + '_input')
        assert_equal(f('a'), ord('a'))
        assert_equal(f(b'a'), ord(b'a'))
        assert_equal(f(''), 0)
        assert_equal(f(b''), 0)
        assert_equal(f(b'\x00'), 0)
        assert_equal(f('ab'), ord('a'))
        assert_equal(f(b'ab'), ord('a'))
        assert_equal(f(['a']), ord('a'))
        assert_equal(f(np.array(b'a')), ord('a'))
        assert_equal(f(np.array([b'a'])), ord('a'))
        a = np.array('a')
        assert_equal(f(a), ord('a'))
        a = np.array(['a'])
        assert_equal(f(a), ord('a'))
        try:
            f([])
        except IndexError as msg:
            if not str(msg).endswith(' got 0-list'):
                raise
        else:
            raise SystemError(f'{f.__name__} should have failed on empty list')
        try:
            f(97)
        except TypeError as msg:
            if not str(msg).endswith(' got int instance'):
                raise
        else:
            raise SystemError(f'{f.__name__} should have failed on int value')

    @pytest.mark.parametrize('dtype', ['c', 'S1', 'U1'])
    def test_array_input(self, dtype):
        f = getattr(self.module, self.fprefix + '_array_input')
        assert_array_equal(f(np.array(['a', 'b', 'c'], dtype=dtype)), np.array(list(map(ord, 'abc')), dtype='i1'))
        assert_array_equal(f(np.array([b'a', b'b', b'c'], dtype=dtype)), np.array(list(map(ord, 'abc')), dtype='i1'))

    def test_array_input_varia(self):
        f = getattr(self.module, self.fprefix + '_array_input')
        assert_array_equal(f(['a', 'b', 'c']), np.array(list(map(ord, 'abc')), dtype='i1'))
        assert_array_equal(f([b'a', b'b', b'c']), np.array(list(map(ord, 'abc')), dtype='i1'))
        try:
            f(['a', 'b', 'c', 'd'])
        except ValueError as msg:
            if not str(msg).endswith('th dimension must be fixed to 3 but got 4'):
                raise
        else:
            raise SystemError(f'{f.__name__} should have failed on wrong input')

    @pytest.mark.parametrize('dtype', ['c', 'S1', 'U1'])
    def test_2d_array_input(self, dtype):
        f = getattr(self.module, self.fprefix + '_2d_array_input')
        a = np.array([['a', 'b', 'c'], ['d', 'e', 'f']], dtype=dtype, order='F')
        expected = a.view(np.uint32 if dtype == 'U1' else np.uint8)
        assert_array_equal(f(a), expected)

    def test_output(self):
        f = getattr(self.module, self.fprefix + '_output')
        assert_equal(f(ord(b'a')), b'a')
        assert_equal(f(0), b'\x00')

    def test_array_output(self):
        f = getattr(self.module, self.fprefix + '_array_output')
        assert_array_equal(f(list(map(ord, 'abc'))), np.array(list('abc'), dtype='S1'))

    def test_input_output(self):
        f = getattr(self.module, self.fprefix + '_input_output')
        assert_equal(f(b'a'), b'a')
        assert_equal(f('a'), b'a')
        assert_equal(f(''), b'\x00')

    @pytest.mark.parametrize('dtype', ['c', 'S1'])
    def test_inout(self, dtype):
        f = getattr(self.module, self.fprefix + '_inout')
        a = np.array(list('abc'), dtype=dtype)
        f(a, 'A')
        assert_array_equal(a, np.array(list('Abc'), dtype=a.dtype))
        f(a[1:], 'B')
        assert_array_equal(a, np.array(list('ABc'), dtype=a.dtype))
        a = np.array(['abc'], dtype=dtype)
        f(a, 'A')
        assert_array_equal(a, np.array(['Abc'], dtype=a.dtype))

    def test_inout_varia(self):
        f = getattr(self.module, self.fprefix + '_inout')
        a = np.array('abc', dtype='S3')
        f(a, 'A')
        assert_array_equal(a, np.array('Abc', dtype=a.dtype))
        a = np.array(['abc'], dtype='S3')
        f(a, 'A')
        assert_array_equal(a, np.array(['Abc'], dtype=a.dtype))
        try:
            f('abc', 'A')
        except ValueError as msg:
            if not str(msg).endswith(' got 3-str'):
                raise
        else:
            raise SystemError(f'{f.__name__} should have failed on str value')

    @pytest.mark.parametrize('dtype', ['c', 'S1'])
    def test_array_inout(self, dtype):
        f = getattr(self.module, self.fprefix + '_array_inout')
        n = np.array(['A', 'B', 'C'], dtype=dtype, order='F')
        a = np.array(['a', 'b', 'c'], dtype=dtype, order='F')
        f(a, n)
        assert_array_equal(a, n)
        a = np.array(['a', 'b', 'c', 'd'], dtype=dtype)
        f(a[1:], n)
        assert_array_equal(a, np.array(['a', 'A', 'B', 'C'], dtype=dtype))
        a = np.array([['a', 'b', 'c']], dtype=dtype, order='F')
        f(a, n)
        assert_array_equal(a, np.array([['A', 'B', 'C']], dtype=dtype))
        a = np.array(['a', 'b', 'c', 'd'], dtype=dtype, order='F')
        try:
            f(a, n)
        except ValueError as msg:
            if not str(msg).endswith('th dimension must be fixed to 3 but got 4'):
                raise
        else:
            raise SystemError(f'{f.__name__} should have failed on wrong input')

    @pytest.mark.parametrize('dtype', ['c', 'S1'])
    def test_2d_array_inout(self, dtype):
        f = getattr(self.module, self.fprefix + '_2d_array_inout')
        n = np.array([['A', 'B', 'C'], ['D', 'E', 'F']], dtype=dtype, order='F')
        a = np.array([['a', 'b', 'c'], ['d', 'e', 'f']], dtype=dtype, order='F')
        f(a, n)
        assert_array_equal(a, n)

    def test_return(self):
        f = getattr(self.module, self.fprefix + '_return')
        assert_equal(f('a'), b'a')

    @pytest.mark.skip('fortran function returning array segfaults')
    def test_array_return(self):
        f = getattr(self.module, self.fprefix + '_array_return')
        a = np.array(list('abc'), dtype='S1')
        assert_array_equal(f(a), a)

    def test_optional(self):
        f = getattr(self.module, self.fprefix + '_optional')
        assert_equal(f(), b'a')
        assert_equal(f(b'B'), b'B')