from ..Qt import QtWidgets
def addLayout(self, row=None, col=None, rowspan=1, colspan=1, **kargs):
    """
        Create an empty LayoutWidget and place it in the next available cell (or in the cell specified)
        All extra keyword arguments are passed to :func:`LayoutWidget.__init__ <pyqtgraph.LayoutWidget.__init__>`
        Returns the created widget.
        """
    layout = LayoutWidget(**kargs)
    self.addWidget(layout, row, col, rowspan, colspan)
    return layout