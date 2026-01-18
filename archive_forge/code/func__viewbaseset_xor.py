from collections.abc import ItemsView, Iterable, KeysView, Set, ValuesView
def _viewbaseset_xor(view, other):
    if not isinstance(other, Iterable):
        return NotImplemented
    if isinstance(view, Set):
        view = set(iter(view))
    if isinstance(other, Set):
        other = set(iter(other))
    if not isinstance(other, Set):
        other = set(iter(other))
    return view ^ other