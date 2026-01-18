import re
import sys
from docutils import DataError
from docutils.utils import strip_combining_chars
class GridTableParser(TableParser):
    """
    Parse a grid table using `parse()`.

    Here's an example of a grid table::

        +------------------------+------------+----------+----------+
        | Header row, column 1   | Header 2   | Header 3 | Header 4 |
        +========================+============+==========+==========+
        | body row 1, column 1   | column 2   | column 3 | column 4 |
        +------------------------+------------+----------+----------+
        | body row 2             | Cells may span columns.          |
        +------------------------+------------+---------------------+
        | body row 3             | Cells may  | - Table cells       |
        +------------------------+ span rows. | - contain           |
        | body row 4             |            | - body elements.    |
        +------------------------+------------+---------------------+

    Intersections use '+', row separators use '-' (except for one optional
    head/body row separator, which uses '='), and column separators use '|'.

    Passing the above table to the `parse()` method will result in the
    following data structure::

        ([24, 12, 10, 10],
         [[(0, 0, 1, ['Header row, column 1']),
           (0, 0, 1, ['Header 2']),
           (0, 0, 1, ['Header 3']),
           (0, 0, 1, ['Header 4'])]],
         [[(0, 0, 3, ['body row 1, column 1']),
           (0, 0, 3, ['column 2']),
           (0, 0, 3, ['column 3']),
           (0, 0, 3, ['column 4'])],
          [(0, 0, 5, ['body row 2']),
           (0, 2, 5, ['Cells may span columns.']),
           None,
           None],
          [(0, 0, 7, ['body row 3']),
           (1, 0, 7, ['Cells may', 'span rows.', '']),
           (1, 1, 7, ['- Table cells', '- contain', '- body elements.']),
           None],
          [(0, 0, 9, ['body row 4']), None, None, None]])

    The first item is a list containing column widths (colspecs). The second
    item is a list of head rows, and the third is a list of body rows. Each
    row contains a list of cells. Each cell is either None (for a cell unused
    because of another cell's span), or a tuple. A cell tuple contains four
    items: the number of extra rows used by the cell in a vertical span
    (morerows); the number of extra columns used by the cell in a horizontal
    span (morecols); the line offset of the first line of the cell contents;
    and the cell contents, a list of lines of text.
    """
    head_body_separator_pat = re.compile('\\+=[=+]+=\\+ *$')

    def setup(self, block):
        self.block = block[:]
        self.block.disconnect()
        self.bottom = len(block) - 1
        self.right = len(block[0]) - 1
        self.head_body_sep = None
        self.done = [-1] * len(block[0])
        self.cells = []
        self.rowseps = {0: [0]}
        self.colseps = {0: [0]}

    def parse_table(self):
        """
        Start with a queue of upper-left corners, containing the upper-left
        corner of the table itself. Trace out one rectangular cell, remember
        it, and add its upper-right and lower-left corners to the queue of
        potential upper-left corners of further cells. Process the queue in
        top-to-bottom order, keeping track of how much of each text column has
        been seen.

        We'll end up knowing all the row and column boundaries, cell positions
        and their dimensions.
        """
        corners = [(0, 0)]
        while corners:
            top, left = corners.pop(0)
            if top == self.bottom or left == self.right or top <= self.done[left]:
                continue
            result = self.scan_cell(top, left)
            if not result:
                continue
            bottom, right, rowseps, colseps = result
            update_dict_of_lists(self.rowseps, rowseps)
            update_dict_of_lists(self.colseps, colseps)
            self.mark_done(top, left, bottom, right)
            cellblock = self.block.get_2D_block(top + 1, left + 1, bottom, right)
            cellblock.disconnect()
            cellblock.replace(self.double_width_pad_char, '')
            self.cells.append((top, left, bottom, right, cellblock))
            corners.extend([(top, right), (bottom, left)])
            corners.sort()
        if not self.check_parse_complete():
            raise TableMarkupError('Malformed table; parse incomplete.')

    def mark_done(self, top, left, bottom, right):
        """For keeping track of how much of each text column has been seen."""
        before = top - 1
        after = bottom - 1
        for col in range(left, right):
            assert self.done[col] == before
            self.done[col] = after

    def check_parse_complete(self):
        """Each text column should have been completely seen."""
        last = self.bottom - 1
        for col in range(self.right):
            if self.done[col] != last:
                return False
        return True

    def scan_cell(self, top, left):
        """Starting at the top-left corner, start tracing out a cell."""
        assert self.block[top][left] == '+'
        result = self.scan_right(top, left)
        return result

    def scan_right(self, top, left):
        """
        Look for the top-right corner of the cell, and make note of all column
        boundaries ('+').
        """
        colseps = {}
        line = self.block[top]
        for i in range(left + 1, self.right + 1):
            if line[i] == '+':
                colseps[i] = [top]
                result = self.scan_down(top, left, i)
                if result:
                    bottom, rowseps, newcolseps = result
                    update_dict_of_lists(colseps, newcolseps)
                    return (bottom, i, rowseps, colseps)
            elif line[i] != '-':
                return None
        return None

    def scan_down(self, top, left, right):
        """
        Look for the bottom-right corner of the cell, making note of all row
        boundaries.
        """
        rowseps = {}
        for i in range(top + 1, self.bottom + 1):
            if self.block[i][right] == '+':
                rowseps[i] = [right]
                result = self.scan_left(top, left, i, right)
                if result:
                    newrowseps, colseps = result
                    update_dict_of_lists(rowseps, newrowseps)
                    return (i, rowseps, colseps)
            elif self.block[i][right] != '|':
                return None
        return None

    def scan_left(self, top, left, bottom, right):
        """
        Noting column boundaries, look for the bottom-left corner of the cell.
        It must line up with the starting point.
        """
        colseps = {}
        line = self.block[bottom]
        for i in range(right - 1, left, -1):
            if line[i] == '+':
                colseps[i] = [bottom]
            elif line[i] != '-':
                return None
        if line[left] != '+':
            return None
        result = self.scan_up(top, left, bottom, right)
        if result is not None:
            rowseps = result
            return (rowseps, colseps)
        return None

    def scan_up(self, top, left, bottom, right):
        """
        Noting row boundaries, see if we can return to the starting point.
        """
        rowseps = {}
        for i in range(bottom - 1, top, -1):
            if self.block[i][left] == '+':
                rowseps[i] = [left]
            elif self.block[i][left] != '|':
                return None
        return rowseps

    def structure_from_cells(self):
        """
        From the data collected by `scan_cell()`, convert to the final data
        structure.
        """
        rowseps = list(self.rowseps.keys())
        rowseps.sort()
        rowindex = {}
        for i in range(len(rowseps)):
            rowindex[rowseps[i]] = i
        colseps = list(self.colseps.keys())
        colseps.sort()
        colindex = {}
        for i in range(len(colseps)):
            colindex[colseps[i]] = i
        colspecs = [colseps[i] - colseps[i - 1] - 1 for i in range(1, len(colseps))]
        onerow = [None for i in range(len(colseps) - 1)]
        rows = [onerow[:] for i in range(len(rowseps) - 1)]
        remaining = (len(rowseps) - 1) * (len(colseps) - 1)
        for top, left, bottom, right, block in self.cells:
            rownum = rowindex[top]
            colnum = colindex[left]
            assert rows[rownum][colnum] is None, 'Cell (row %s, column %s) already used.' % (rownum + 1, colnum + 1)
            morerows = rowindex[bottom] - rownum - 1
            morecols = colindex[right] - colnum - 1
            remaining -= (morerows + 1) * (morecols + 1)
            rows[rownum][colnum] = (morerows, morecols, top + 1, block)
        assert remaining == 0, 'Unused cells remaining.'
        if self.head_body_sep:
            numheadrows = rowindex[self.head_body_sep]
            headrows = rows[:numheadrows]
            bodyrows = rows[numheadrows:]
        else:
            headrows = []
            bodyrows = rows
        return (colspecs, headrows, bodyrows)