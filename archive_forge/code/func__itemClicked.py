from ..Qt import QtCore, QtWidgets
def _itemClicked(self, item, col):
    if hasattr(item, 'itemClicked'):
        item.itemClicked(col)