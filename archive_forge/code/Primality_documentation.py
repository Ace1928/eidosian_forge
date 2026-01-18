from Cryptodome import Random
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Util.py3compat import iter_range
from Cryptodome.Util.number import sieve_base as _sieve_base_large
Generate a random, probable safe prime.

    Note this operation is much slower than generating a simple prime.

    :Keywords:
      exact_bits : integer
        The desired size in bits of the probable safe prime.
      randfunc : callable
        An RNG function where candidate primes are taken from.

    :Return:
        A probable safe prime in the range
        2^exact_bits > p > 2^(exact_bits-1).
    