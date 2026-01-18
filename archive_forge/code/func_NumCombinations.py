import itertools
def NumCombinations(nItems, nSlots):
    """  returns the number of ways to fit nItems into nSlots

      We assume that (x, y) and (y, x) are equivalent, and
      (x, x) is allowed.

      General formula is, for N items and S slots:
        res = (N+S-1)! / ( (N-1)! * S! )

    """
    global _numCombDict
    res = _numCombDict.get((nItems, nSlots), -1)
    if res == -1:
        res = comb(nItems + nSlots - 1, nSlots)
        _numCombDict[nItems, nSlots] = res
    return res