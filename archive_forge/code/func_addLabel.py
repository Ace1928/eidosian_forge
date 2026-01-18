from ..Qt import QtWidgets
def addLabel(self, text=' ', row=None, col=None, rowspan=1, colspan=1, **kargs):
    """
        Create a QLabel with *text* and place it in the next available cell (or in the cell specified)
        All extra keyword arguments are passed to QLabel().
        Returns the created widget.
        """
    text = QtWidgets.QLabel(text, **kargs)
    self.addWidget(text, row, col, rowspan, colspan)
    return text