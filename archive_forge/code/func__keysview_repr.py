from collections.abc import ItemsView, Iterable, KeysView, Set, ValuesView
def _keysview_repr(view):
    lst = []
    for k in view:
        lst.append('{!r}'.format(k))
    body = ', '.join(lst)
    return '{}({})'.format(view.__class__.__name__, body)