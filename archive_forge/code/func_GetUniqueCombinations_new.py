import itertools
def GetUniqueCombinations_new(choices, classes, which=0):
    """  Does the combinatorial explosion of the possible combinations
    of the elements of _choices_.

    """
    assert len(choices) == len(classes)
    combos = set()
    for choice in itertools.product(*choices):
        if len(set(choice)) != len(choice):
            continue
        combos.add(tuple(sorted(((cls, ch) for cls, ch in zip(classes, choice)))))
    return [list(combo) for combo in sorted(combos)]