import threading
import typing
import warnings
import rsa.prime
import rsa.pem
import rsa.common
import rsa.randnum
import rsa.core
def _initial_blinding_factor(self) -> int:
    for _ in range(1000):
        blind_r = rsa.randnum.randint(self.n - 1)
        if rsa.prime.are_relatively_prime(self.n, blind_r):
            return blind_r
    raise RuntimeError('unable to find blinding factor')