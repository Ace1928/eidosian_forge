import threading
import typing
import warnings
import rsa.prime
import rsa.pem
import rsa.common
import rsa.randnum
import rsa.core
def _update_blinding_factor(self) -> typing.Tuple[int, int]:
    """Update blinding factors.

        Computing a blinding factor is expensive, so instead this function
        does this once, then updates the blinding factor as per section 9
        of 'A Timing Attack against RSA with the Chinese Remainder Theorem'
        by Werner Schindler.
        See https://tls.mbed.org/public/WSchindler-RSA_Timing_Attack.pdf

        :return: the new blinding factor and its inverse.
        """
    with self.mutex:
        if self.blindfac < 0:
            self.blindfac = self._initial_blinding_factor()
            self.blindfac_inverse = rsa.common.inverse(self.blindfac, self.n)
        else:
            self.blindfac = pow(self.blindfac, 2, self.n)
            self.blindfac_inverse = pow(self.blindfac_inverse, 2, self.n)
        return (self.blindfac, self.blindfac_inverse)