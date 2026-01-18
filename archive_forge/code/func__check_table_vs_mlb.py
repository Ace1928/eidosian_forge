import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk
def _check_table_vs_mlb(self):
    """
        Verify that the contents of the table's ``_rows`` variable match
        the contents of its multi-listbox (``_mlb``).  This is just
        included for debugging purposes, to make sure that the
        list-modifying operations are working correctly.
        """
    for col in self._mlb.listboxes:
        assert len(self) == col.size()
    for row in self:
        assert len(row) == self._num_columns
    assert self._num_columns == len(self._mlb.column_names)
    for i, row in enumerate(self):
        for j, cell in enumerate(row):
            if self._reprfunc is not None:
                cell = self._reprfunc(i, j, cell)
            assert self._mlb.get(i)[j] == cell