from .knit import KnitCorrupt
def _load_data_py(kndx, fp):
    """Read in a knit index."""
    cache = kndx._cache
    history = kndx._history
    kndx.check_header(fp)
    history_top = len(history) - 1
    for line in fp.readlines():
        rec = line.split()
        if len(rec) < 5 or rec[-1] != b':':
            continue
        try:
            parents = []
            for value in rec[4:-1]:
                if value[:1] == b'.':
                    parent_id = value[1:]
                else:
                    parent_id = history[int(value)]
                parents.append(parent_id)
        except (IndexError, ValueError) as e:
            raise KnitCorrupt(kndx._filename, 'line {!r}: {}'.format(rec, e))
        version_id, options, pos, size = rec[:4]
        try:
            pos = int(pos)
        except ValueError as e:
            raise KnitCorrupt(kndx._filename, 'invalid position on line %r: %s' % (rec, e))
        try:
            size = int(size)
        except ValueError as e:
            raise KnitCorrupt(kndx._filename, 'invalid size on line %r: %s' % (rec, e))
        if version_id not in cache:
            history_top += 1
            index = history_top
            history.append(version_id)
        else:
            index = cache[version_id][5]
        cache[version_id] = (version_id, options.split(b','), pos, size, tuple(parents), index)