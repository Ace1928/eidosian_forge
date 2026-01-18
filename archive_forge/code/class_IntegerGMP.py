import sys
from Cryptodome.Util.py3compat import tobytes, is_native_int
from Cryptodome.Util._raw_api import (backend, load_lib,
from ._IntegerBase import IntegerBase
class IntegerGMP(IntegerBase):
    """A fast, arbitrary precision integer"""
    _zero_mpz_p = new_mpz()
    _gmp.mpz_init_set_ui(_zero_mpz_p, c_ulong(0))

    def __init__(self, value):
        """Initialize the integer to the given value."""
        self._mpz_p = new_mpz()
        self._initialized = False
        if isinstance(value, float):
            raise ValueError('A floating point type is not a natural number')
        if is_native_int(value):
            _gmp.mpz_init(self._mpz_p)
            self._initialized = True
            if value == 0:
                return
            tmp = new_mpz()
            _gmp.mpz_init(tmp)
            try:
                positive = value >= 0
                reduce = abs(value)
                slots = (reduce.bit_length() - 1) // 32 + 1
                while slots > 0:
                    slots = slots - 1
                    _gmp.mpz_set_ui(tmp, c_ulong(4294967295 & reduce >> slots * 32))
                    _gmp.mpz_mul_2exp(tmp, tmp, c_ulong(slots * 32))
                    _gmp.mpz_add(self._mpz_p, self._mpz_p, tmp)
            finally:
                _gmp.mpz_clear(tmp)
            if not positive:
                _gmp.mpz_neg(self._mpz_p, self._mpz_p)
        elif isinstance(value, IntegerGMP):
            _gmp.mpz_init_set(self._mpz_p, value._mpz_p)
            self._initialized = True
        else:
            raise NotImplementedError

    def __int__(self):
        tmp = new_mpz()
        _gmp.mpz_init_set(tmp, self._mpz_p)
        try:
            value = 0
            slot = 0
            while _gmp.mpz_cmp(tmp, self._zero_mpz_p) != 0:
                lsb = _gmp.mpz_get_ui(tmp) & 4294967295
                value |= lsb << slot * 32
                _gmp.mpz_tdiv_q_2exp(tmp, tmp, c_ulong(32))
                slot = slot + 1
        finally:
            _gmp.mpz_clear(tmp)
        if self < 0:
            value = -value
        return int(value)

    def __str__(self):
        return str(int(self))

    def __repr__(self):
        return 'Integer(%s)' % str(self)

    def __hex__(self):
        return hex(int(self))

    def __index__(self):
        return int(self)

    def to_bytes(self, block_size=0, byteorder='big'):
        """Convert the number into a byte string.

        This method encodes the number in network order and prepends
        as many zero bytes as required. It only works for non-negative
        values.

        :Parameters:
          block_size : integer
            The exact size the output byte string must have.
            If zero, the string has the minimal length.
          byteorder : string
            'big' for big-endian integers (default), 'little' for litte-endian.
        :Returns:
          A byte string.
        :Raise ValueError:
          If the value is negative or if ``block_size`` is
          provided and the length of the byte string would exceed it.
        """
        if self < 0:
            raise ValueError('Conversion only valid for non-negative numbers')
        buf_len = (_gmp.mpz_sizeinbase(self._mpz_p, 2) + 7) // 8
        if buf_len > block_size > 0:
            raise ValueError('Number is too big to convert to byte string of prescribed length')
        buf = create_string_buffer(buf_len)
        _gmp.mpz_export(buf, null_pointer, 1, c_size_t(1), 0, c_size_t(0), self._mpz_p)
        result = b'\x00' * max(0, block_size - buf_len) + get_raw_buffer(buf)
        if byteorder == 'big':
            pass
        elif byteorder == 'little':
            result = bytearray(result)
            result.reverse()
            result = bytes(result)
        else:
            raise ValueError('Incorrect byteorder')
        return result

    @staticmethod
    def from_bytes(byte_string, byteorder='big'):
        """Convert a byte string into a number.

        :Parameters:
          byte_string : byte string
            The input number, encoded in network order.
            It can only be non-negative.
          byteorder : string
            'big' for big-endian integers (default), 'little' for litte-endian.

        :Return:
          The ``Integer`` object carrying the same value as the input.
        """
        result = IntegerGMP(0)
        if byteorder == 'big':
            pass
        elif byteorder == 'little':
            byte_string = bytearray(byte_string)
            byte_string.reverse()
        else:
            raise ValueError('Incorrect byteorder')
        _gmp.mpz_import(result._mpz_p, c_size_t(len(byte_string)), 1, c_size_t(1), 0, c_size_t(0), c_uint8_ptr(byte_string))
        return result

    def _apply_and_return(self, func, term):
        if not isinstance(term, IntegerGMP):
            term = IntegerGMP(term)
        return func(self._mpz_p, term._mpz_p)

    def __eq__(self, term):
        if not (isinstance(term, IntegerGMP) or is_native_int(term)):
            return False
        return self._apply_and_return(_gmp.mpz_cmp, term) == 0

    def __ne__(self, term):
        if not (isinstance(term, IntegerGMP) or is_native_int(term)):
            return True
        return self._apply_and_return(_gmp.mpz_cmp, term) != 0

    def __lt__(self, term):
        return self._apply_and_return(_gmp.mpz_cmp, term) < 0

    def __le__(self, term):
        return self._apply_and_return(_gmp.mpz_cmp, term) <= 0

    def __gt__(self, term):
        return self._apply_and_return(_gmp.mpz_cmp, term) > 0

    def __ge__(self, term):
        return self._apply_and_return(_gmp.mpz_cmp, term) >= 0

    def __nonzero__(self):
        return _gmp.mpz_cmp(self._mpz_p, self._zero_mpz_p) != 0
    __bool__ = __nonzero__

    def is_negative(self):
        return _gmp.mpz_cmp(self._mpz_p, self._zero_mpz_p) < 0

    def __add__(self, term):
        result = IntegerGMP(0)
        if not isinstance(term, IntegerGMP):
            try:
                term = IntegerGMP(term)
            except NotImplementedError:
                return NotImplemented
        _gmp.mpz_add(result._mpz_p, self._mpz_p, term._mpz_p)
        return result

    def __sub__(self, term):
        result = IntegerGMP(0)
        if not isinstance(term, IntegerGMP):
            try:
                term = IntegerGMP(term)
            except NotImplementedError:
                return NotImplemented
        _gmp.mpz_sub(result._mpz_p, self._mpz_p, term._mpz_p)
        return result

    def __mul__(self, term):
        result = IntegerGMP(0)
        if not isinstance(term, IntegerGMP):
            try:
                term = IntegerGMP(term)
            except NotImplementedError:
                return NotImplemented
        _gmp.mpz_mul(result._mpz_p, self._mpz_p, term._mpz_p)
        return result

    def __floordiv__(self, divisor):
        if not isinstance(divisor, IntegerGMP):
            divisor = IntegerGMP(divisor)
        if _gmp.mpz_cmp(divisor._mpz_p, self._zero_mpz_p) == 0:
            raise ZeroDivisionError('Division by zero')
        result = IntegerGMP(0)
        _gmp.mpz_fdiv_q(result._mpz_p, self._mpz_p, divisor._mpz_p)
        return result

    def __mod__(self, divisor):
        if not isinstance(divisor, IntegerGMP):
            divisor = IntegerGMP(divisor)
        comp = _gmp.mpz_cmp(divisor._mpz_p, self._zero_mpz_p)
        if comp == 0:
            raise ZeroDivisionError('Division by zero')
        if comp < 0:
            raise ValueError('Modulus must be positive')
        result = IntegerGMP(0)
        _gmp.mpz_mod(result._mpz_p, self._mpz_p, divisor._mpz_p)
        return result

    def inplace_pow(self, exponent, modulus=None):
        if modulus is None:
            if exponent < 0:
                raise ValueError('Exponent must not be negative')
            if exponent > 256:
                raise ValueError('Exponent is too big')
            _gmp.mpz_pow_ui(self._mpz_p, self._mpz_p, c_ulong(int(exponent)))
        else:
            if not isinstance(modulus, IntegerGMP):
                modulus = IntegerGMP(modulus)
            if not modulus:
                raise ZeroDivisionError('Division by zero')
            if modulus.is_negative():
                raise ValueError('Modulus must be positive')
            if is_native_int(exponent):
                if exponent < 0:
                    raise ValueError('Exponent must not be negative')
                if exponent < 65536:
                    _gmp.mpz_powm_ui(self._mpz_p, self._mpz_p, c_ulong(exponent), modulus._mpz_p)
                    return self
                exponent = IntegerGMP(exponent)
            elif exponent.is_negative():
                raise ValueError('Exponent must not be negative')
            _gmp.mpz_powm(self._mpz_p, self._mpz_p, exponent._mpz_p, modulus._mpz_p)
        return self

    def __pow__(self, exponent, modulus=None):
        result = IntegerGMP(self)
        return result.inplace_pow(exponent, modulus)

    def __abs__(self):
        result = IntegerGMP(0)
        _gmp.mpz_abs(result._mpz_p, self._mpz_p)
        return result

    def sqrt(self, modulus=None):
        """Return the largest Integer that does not
        exceed the square root"""
        if modulus is None:
            if self < 0:
                raise ValueError('Square root of negative value')
            result = IntegerGMP(0)
            _gmp.mpz_sqrt(result._mpz_p, self._mpz_p)
        else:
            if modulus <= 0:
                raise ValueError('Modulus must be positive')
            modulus = int(modulus)
            result = IntegerGMP(self._tonelli_shanks(int(self) % modulus, modulus))
        return result

    def __iadd__(self, term):
        if is_native_int(term):
            if 0 <= term < 65536:
                _gmp.mpz_add_ui(self._mpz_p, self._mpz_p, c_ulong(term))
                return self
            if -65535 < term < 0:
                _gmp.mpz_sub_ui(self._mpz_p, self._mpz_p, c_ulong(-term))
                return self
            term = IntegerGMP(term)
        _gmp.mpz_add(self._mpz_p, self._mpz_p, term._mpz_p)
        return self

    def __isub__(self, term):
        if is_native_int(term):
            if 0 <= term < 65536:
                _gmp.mpz_sub_ui(self._mpz_p, self._mpz_p, c_ulong(term))
                return self
            if -65535 < term < 0:
                _gmp.mpz_add_ui(self._mpz_p, self._mpz_p, c_ulong(-term))
                return self
            term = IntegerGMP(term)
        _gmp.mpz_sub(self._mpz_p, self._mpz_p, term._mpz_p)
        return self

    def __imul__(self, term):
        if is_native_int(term):
            if 0 <= term < 65536:
                _gmp.mpz_mul_ui(self._mpz_p, self._mpz_p, c_ulong(term))
                return self
            if -65535 < term < 0:
                _gmp.mpz_mul_ui(self._mpz_p, self._mpz_p, c_ulong(-term))
                _gmp.mpz_neg(self._mpz_p, self._mpz_p)
                return self
            term = IntegerGMP(term)
        _gmp.mpz_mul(self._mpz_p, self._mpz_p, term._mpz_p)
        return self

    def __imod__(self, divisor):
        if not isinstance(divisor, IntegerGMP):
            divisor = IntegerGMP(divisor)
        comp = _gmp.mpz_cmp(divisor._mpz_p, divisor._zero_mpz_p)
        if comp == 0:
            raise ZeroDivisionError('Division by zero')
        if comp < 0:
            raise ValueError('Modulus must be positive')
        _gmp.mpz_mod(self._mpz_p, self._mpz_p, divisor._mpz_p)
        return self

    def __and__(self, term):
        result = IntegerGMP(0)
        if not isinstance(term, IntegerGMP):
            term = IntegerGMP(term)
        _gmp.mpz_and(result._mpz_p, self._mpz_p, term._mpz_p)
        return result

    def __or__(self, term):
        result = IntegerGMP(0)
        if not isinstance(term, IntegerGMP):
            term = IntegerGMP(term)
        _gmp.mpz_ior(result._mpz_p, self._mpz_p, term._mpz_p)
        return result

    def __rshift__(self, pos):
        result = IntegerGMP(0)
        if pos < 0:
            raise ValueError('negative shift count')
        if pos > 65536:
            if self < 0:
                return -1
            else:
                return 0
        _gmp.mpz_tdiv_q_2exp(result._mpz_p, self._mpz_p, c_ulong(int(pos)))
        return result

    def __irshift__(self, pos):
        if pos < 0:
            raise ValueError('negative shift count')
        if pos > 65536:
            if self < 0:
                return -1
            else:
                return 0
        _gmp.mpz_tdiv_q_2exp(self._mpz_p, self._mpz_p, c_ulong(int(pos)))
        return self

    def __lshift__(self, pos):
        result = IntegerGMP(0)
        if not 0 <= pos < 65536:
            raise ValueError('Incorrect shift count')
        _gmp.mpz_mul_2exp(result._mpz_p, self._mpz_p, c_ulong(int(pos)))
        return result

    def __ilshift__(self, pos):
        if not 0 <= pos < 65536:
            raise ValueError('Incorrect shift count')
        _gmp.mpz_mul_2exp(self._mpz_p, self._mpz_p, c_ulong(int(pos)))
        return self

    def get_bit(self, n):
        """Return True if the n-th bit is set to 1.
        Bit 0 is the least significant."""
        if self < 0:
            raise ValueError('no bit representation for negative values')
        if n < 0:
            raise ValueError('negative bit count')
        if n > 65536:
            return 0
        return bool(_gmp.mpz_tstbit(self._mpz_p, c_ulong(int(n))))

    def is_odd(self):
        return _gmp.mpz_tstbit(self._mpz_p, 0) == 1

    def is_even(self):
        return _gmp.mpz_tstbit(self._mpz_p, 0) == 0

    def size_in_bits(self):
        """Return the minimum number of bits that can encode the number."""
        if self < 0:
            raise ValueError('Conversion only valid for non-negative numbers')
        return _gmp.mpz_sizeinbase(self._mpz_p, 2)

    def size_in_bytes(self):
        """Return the minimum number of bytes that can encode the number."""
        return (self.size_in_bits() - 1) // 8 + 1

    def is_perfect_square(self):
        return _gmp.mpz_perfect_square_p(self._mpz_p) != 0

    def fail_if_divisible_by(self, small_prime):
        """Raise an exception if the small prime is a divisor."""
        if is_native_int(small_prime):
            if 0 < small_prime < 65536:
                if _gmp.mpz_divisible_ui_p(self._mpz_p, c_ulong(small_prime)):
                    raise ValueError('The value is composite')
                return
            small_prime = IntegerGMP(small_prime)
        if _gmp.mpz_divisible_p(self._mpz_p, small_prime._mpz_p):
            raise ValueError('The value is composite')

    def multiply_accumulate(self, a, b):
        """Increment the number by the product of a and b."""
        if not isinstance(a, IntegerGMP):
            a = IntegerGMP(a)
        if is_native_int(b):
            if 0 < b < 65536:
                _gmp.mpz_addmul_ui(self._mpz_p, a._mpz_p, c_ulong(b))
                return self
            if -65535 < b < 0:
                _gmp.mpz_submul_ui(self._mpz_p, a._mpz_p, c_ulong(-b))
                return self
            b = IntegerGMP(b)
        _gmp.mpz_addmul(self._mpz_p, a._mpz_p, b._mpz_p)
        return self

    def set(self, source):
        """Set the Integer to have the given value"""
        if not isinstance(source, IntegerGMP):
            source = IntegerGMP(source)
        _gmp.mpz_set(self._mpz_p, source._mpz_p)
        return self

    def inplace_inverse(self, modulus):
        """Compute the inverse of this number in the ring of
        modulo integers.

        Raise an exception if no inverse exists.
        """
        if not isinstance(modulus, IntegerGMP):
            modulus = IntegerGMP(modulus)
        comp = _gmp.mpz_cmp(modulus._mpz_p, self._zero_mpz_p)
        if comp == 0:
            raise ZeroDivisionError('Modulus cannot be zero')
        if comp < 0:
            raise ValueError('Modulus must be positive')
        result = _gmp.mpz_invert(self._mpz_p, self._mpz_p, modulus._mpz_p)
        if not result:
            raise ValueError('No inverse value can be computed')
        return self

    def inverse(self, modulus):
        result = IntegerGMP(self)
        result.inplace_inverse(modulus)
        return result

    def gcd(self, term):
        """Compute the greatest common denominator between this
        number and another term."""
        result = IntegerGMP(0)
        if is_native_int(term):
            if 0 < term < 65535:
                _gmp.mpz_gcd_ui(result._mpz_p, self._mpz_p, c_ulong(term))
                return result
            term = IntegerGMP(term)
        _gmp.mpz_gcd(result._mpz_p, self._mpz_p, term._mpz_p)
        return result

    def lcm(self, term):
        """Compute the least common multiplier between this
        number and another term."""
        result = IntegerGMP(0)
        if not isinstance(term, IntegerGMP):
            term = IntegerGMP(term)
        _gmp.mpz_lcm(result._mpz_p, self._mpz_p, term._mpz_p)
        return result

    @staticmethod
    def jacobi_symbol(a, n):
        """Compute the Jacobi symbol"""
        if not isinstance(a, IntegerGMP):
            a = IntegerGMP(a)
        if not isinstance(n, IntegerGMP):
            n = IntegerGMP(n)
        if n <= 0 or n.is_even():
            raise ValueError('n must be positive odd for the Jacobi symbol')
        return _gmp.mpz_jacobi(a._mpz_p, n._mpz_p)

    @staticmethod
    def _mult_modulo_bytes(term1, term2, modulus):
        if not isinstance(term1, IntegerGMP):
            term1 = IntegerGMP(term1)
        if not isinstance(term2, IntegerGMP):
            term2 = IntegerGMP(term2)
        if not isinstance(modulus, IntegerGMP):
            modulus = IntegerGMP(modulus)
        if modulus < 0:
            raise ValueError('Modulus must be positive')
        if modulus == 0:
            raise ZeroDivisionError('Modulus cannot be zero')
        if modulus & 1 == 0:
            raise ValueError('Odd modulus is required')
        numbers_len = len(modulus.to_bytes())
        result = (term1 * term2 % modulus).to_bytes(numbers_len)
        return result

    def __del__(self):
        try:
            if self._mpz_p is not None:
                if self._initialized:
                    _gmp.mpz_clear(self._mpz_p)
            self._mpz_p = None
        except AttributeError:
            pass