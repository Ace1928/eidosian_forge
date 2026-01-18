import itertools
def GetUniqueCombinations(choices, classes, which=0):
    """  Does the combinatorial explosion of the possible combinations
    of the elements of _choices_.

    """
    assert len(choices) == len(classes)
    if which >= len(choices):
        return []
    if which == len(choices) - 1:
        return [[(classes[which], x)] for x in choices[which]]
    res = []
    tmp = GetUniqueCombinations(choices, classes, which=which + 1)
    for thing in choices[which]:
        for other in tmp:
            if not any((x[1] == thing for x in other)):
                newL = [(classes[which], thing)] + other
                newL.sort()
                if newL not in res:
                    res.append(newL)
    return res