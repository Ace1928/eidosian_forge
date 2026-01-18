import itertools
def GetAllCombinations(choices, noDups=1, which=0):
    """  Does the combinatorial explosion of the possible combinations
    of the elements of _choices_.

    **Arguments**

      - choices: sequence of sequences with the elements to be enumerated

      - noDups: (optional) if this is nonzero, results with duplicates,
        e.g. (1,1,0), will not be generated

      - which: used in recursion

    **Returns**

      a list of lists

    >>> GetAllCombinations([(0, ), (1, ), (2, )])
    [[0, 1, 2]]
    >>> GetAllCombinations([(0, ), (1, 3), (2, )])
    [[0, 1, 2], [0, 3, 2]]

    >>> GetAllCombinations([(0, 1), (1, 3), (2, )])
    [[0, 1, 2], [0, 3, 2], [1, 3, 2]]

    """
    if which >= len(choices):
        return []
    elif which == len(choices) - 1:
        return [[x] for x in choices[which]]
    res = []
    tmp = GetAllCombinations(choices, noDups=noDups, which=which + 1)
    for thing in choices[which]:
        for other in tmp:
            if not noDups or thing not in other:
                res.append([thing] + other)
    return res