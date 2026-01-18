from Cryptodome.Util.py3compat import is_native_int
from Cryptodome.Util import number
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Random import get_random_bytes as rng
class Shamir(object):
    """Shamir's secret sharing scheme.

    A secret is split into ``n`` shares, and it is sufficient to collect
    ``k`` of them to reconstruct the secret.
    """

    @staticmethod
    def split(k, n, secret, ssss=False):
        """Split a secret into ``n`` shares.

        The secret can be reconstructed later using just ``k`` shares
        out of the original ``n``.
        Each share must be kept confidential to the person it was
        assigned to.

        Each share is associated to an index (starting from 1).

        Args:
          k (integer):
            The sufficient number of shares to reconstruct the secret (``k < n``).
          n (integer):
            The number of shares that this method will create.
          secret (byte string):
            A byte string of 16 bytes (e.g. the AES 128 key).
          ssss (bool):
            If ``True``, the shares can be used with the ``ssss`` utility.
            Default: ``False``.

        Return (tuples):
            ``n`` tuples. A tuple is meant for each participant and it contains two items:

            1. the unique index (an integer)
            2. the share (a byte string, 16 bytes)
        """
        coeffs = [_Element(rng(16)) for i in range(k - 1)]
        coeffs.append(_Element(secret))

        def make_share(user, coeffs, ssss):
            idx = _Element(user)
            share = _Element(0)
            for coeff in coeffs:
                share = idx * share + coeff
            if ssss:
                share += _Element(user) ** len(coeffs)
            return share.encode()
        return [(i, make_share(i, coeffs, ssss)) for i in range(1, n + 1)]

    @staticmethod
    def combine(shares, ssss=False):
        """Recombine a secret, if enough shares are presented.

        Args:
          shares (tuples):
            The *k* tuples, each containin the index (an integer) and
            the share (a byte string, 16 bytes long) that were assigned to
            a participant.
          ssss (bool):
            If ``True``, the shares were produced by the ``ssss`` utility.
            Default: ``False``.

        Return:
            The original secret, as a byte string (16 bytes long).
        """
        k = len(shares)
        gf_shares = []
        for x in shares:
            idx = _Element(x[0])
            value = _Element(x[1])
            if any((y[0] == idx for y in gf_shares)):
                raise ValueError('Duplicate share')
            if ssss:
                value += idx ** k
            gf_shares.append((idx, value))
        result = _Element(0)
        for j in range(k):
            x_j, y_j = gf_shares[j]
            numerator = _Element(1)
            denominator = _Element(1)
            for m in range(k):
                x_m = gf_shares[m][0]
                if m != j:
                    numerator *= x_m
                    denominator *= x_j + x_m
            result += y_j * numerator * denominator.inverse()
        return result.encode()