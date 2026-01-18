import sys
def _matchorfail(text, pos):
    match = tokenprog.match(text, pos)
    if match is None:
        raise ValueError(text, pos)
    return (match, match.end())