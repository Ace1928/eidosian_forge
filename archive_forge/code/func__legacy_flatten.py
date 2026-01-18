def _legacy_flatten(begin):
    result = [begin]
    i = 0
    for ob in iter(result):
        i += 1
        result[i:i] = ob.__bases__
    return result