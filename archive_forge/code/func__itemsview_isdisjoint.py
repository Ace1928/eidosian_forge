from collections.abc import ItemsView, Iterable, KeysView, Set, ValuesView
def _itemsview_isdisjoint(view, other):
    """Return True if two sets have a null intersection."""
    for v in other:
        if v in view:
            return False
    return True