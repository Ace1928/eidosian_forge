from collections.abc import ItemsView, Iterable, KeysView, Set, ValuesView
def _mdrepr(md):
    lst = []
    for k, v in md.items():
        lst.append("'{}': {!r}".format(k, v))
    body = ', '.join(lst)
    return '<{}({})>'.format(md.__class__.__name__, body)