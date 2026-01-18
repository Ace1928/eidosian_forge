import math
import warnings
from Bio import SeqUtils, Seq
from Bio import BiopythonWarning
def Tm_Wallace(seq, check=True, strict=True):
    """Calculate and return the Tm using the 'Wallace rule'.

    Tm = 4 degC * (G + C) + 2 degC * (A+T)

    The Wallace rule (Thein & Wallace 1986, in Human genetic diseases: a
    practical approach, 33-50) is often used as rule of thumb for approximate
    Tm calculations for primers of 14 to 20 nt length.

    Non-DNA characters (e.g., E, F, J, !, 1, etc) are ignored by this method.

    Examples:
        >>> from Bio.SeqUtils import MeltingTemp as mt
        >>> mt.Tm_Wallace('ACGTTGCAATGCCGTA')
        48.0
        >>> mt.Tm_Wallace('ACGT TGCA ATGC CGTA')
        48.0
        >>> mt.Tm_Wallace('1ACGT2TGCA3ATGC4CGTA')
        48.0

    """
    seq = str(seq)
    if check:
        seq = _check(seq, 'Tm_Wallace')
    melting_temp = 2 * sum(map(seq.count, ('A', 'T', 'W'))) + 4 * sum(map(seq.count, ('C', 'G', 'S')))
    tmp = 3 * sum(map(seq.count, ('K', 'M', 'N', 'R', 'Y'))) + 10 / 3.0 * sum(map(seq.count, ('B', 'V'))) + 8 / 3.0 * sum(map(seq.count, ('D', 'H')))
    if strict and tmp:
        raise ValueError('ambiguous bases B, D, H, K, M, N, R, V, Y not allowed when strict=True')
    else:
        melting_temp += tmp
    return melting_temp