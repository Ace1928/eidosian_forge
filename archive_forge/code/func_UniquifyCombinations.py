import itertools
def UniquifyCombinations(combos):
    """ uniquifies the combinations in the argument

      **Arguments**:

        - combos: a sequence of sequences

      **Returns**

        - a list of tuples containing the unique combos

    """
    resD = {}
    for combo in combos:
        k = combo[:]
        k.sort()
        resD[tuple(k)] = tuple(combo)
    return list(resD.values())