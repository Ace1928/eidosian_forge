from __future__ import print_function
import multiprocessing as mp
from rsa._compat import range
import rsa.prime
import rsa.randnum
def _find_prime(nbits, pipe):
    while True:
        integer = rsa.randnum.read_random_odd_int(nbits)
        if rsa.prime.is_prime(integer):
            pipe.send(integer)
            return