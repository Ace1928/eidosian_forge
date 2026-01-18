from collections.abc import ItemsView, Iterable, KeysView, Set, ValuesView
def _abc_valuesview_register(view_cls):
    ValuesView.register(view_cls)