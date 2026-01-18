import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def _checkrow(self, rowvalue):
    """
        Helper function: check that a given row value has the correct
        number of elements; and if not, raise an exception.
        """
    if len(rowvalue) != self._num_columns:
        raise ValueError('Row %r has %d columns; expected %d' % (rowvalue, len(rowvalue), self._num_columns))