import math
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
def CosineSimilarity(v1, v2):
    """ Implements the Cosine similarity metric.
     This is the recommended metric in the LaSSI paper

    **Arguments**:

      - two vectors (sequences of bit ids)

    **Returns**: a float.

    **Notes**

      - the vectors must be sorted

    >>> print('%.3f'%CosineSimilarity( (1,2,3,4,10), (2,4,6) ))
    0.516
    >>> print('%.3f'%CosineSimilarity( (1,2,2,3,4), (2,2,4,5,6) ))
    0.714
    >>> print('%.3f'%CosineSimilarity( (1,2,2,3,4), (1,2,2,3,4) ))
    1.000
    >>> print('%.3f'%CosineSimilarity( (1,2,2,3,4), (5,6,7) ))
    0.000
    >>> print('%.3f'%CosineSimilarity( (1,2,2,3,4), () ))
    0.000

    """
    d1 = Dot(v1, v1)
    d2 = Dot(v2, v2)
    denom = math.sqrt(d1 * d2)
    if not denom:
        res = 0.0
    else:
        numer = Dot(v1, v2)
        res = numer / denom
    return res