import struct, warnings
def _entries(attrs, sameval):
    ak = 0
    vals = []
    lastv = 0
    for k, v in attrs:
        if len(vals) and (k != ak + 1 or (sameval and v != lastv)):
            yield (ak - len(vals) + 1, len(vals), vals)
            vals = []
        ak = k
        vals.append(v)
        lastv = v
    yield (ak - len(vals) + 1, len(vals), vals)