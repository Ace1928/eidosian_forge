from collections.abc import ItemsView, Iterable, KeysView, Set, ValuesView
def _itemsview_repr(view):
    lst = []
    for k, v in view:
        lst.append('{!r}: {!r}'.format(k, v))
    body = ', '.join(lst)
    return '{}({})'.format(view.__class__.__name__, body)