import itertools
import copy
import numbers
from Bio.Phylo import BaseTree
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Align import substitution_matrices
class _Matrix:
    """Base class for distance matrix or scoring matrix.

    Accepts a list of names and a lower triangular matrix.::

        matrix = [[0],
                  [1, 0],
                  [2, 3, 0],
                  [4, 5, 6, 0]]
        represents the symmetric matrix of
        [0,1,2,4]
        [1,0,3,5]
        [2,3,0,6]
        [4,5,6,0]

    :Parameters:
        names : list
            names of elements, used for indexing
        matrix : list
            nested list of numerical lists in lower triangular format

    Examples
    --------
    >>> from Bio.Phylo.TreeConstruction import _Matrix
    >>> names = ['Alpha', 'Beta', 'Gamma', 'Delta']
    >>> matrix = [[0], [1, 0], [2, 3, 0], [4, 5, 6, 0]]
    >>> m = _Matrix(names, matrix)
    >>> m
    _Matrix(names=['Alpha', 'Beta', 'Gamma', 'Delta'], matrix=[[0], [1, 0], [2, 3, 0], [4, 5, 6, 0]])

    You can use two indices to get or assign an element in the matrix.

    >>> m[1,2]
    3
    >>> m['Beta','Gamma']
    3
    >>> m['Beta','Gamma'] = 4
    >>> m['Beta','Gamma']
    4

    Further more, you can use one index to get or assign a list of elements related to that index.

    >>> m[0]
    [0, 1, 2, 4]
    >>> m['Alpha']
    [0, 1, 2, 4]
    >>> m['Alpha'] = [0, 7, 8, 9]
    >>> m[0]
    [0, 7, 8, 9]
    >>> m[0,1]
    7

    Also you can delete or insert a column&row of elements by index.

    >>> m
    _Matrix(names=['Alpha', 'Beta', 'Gamma', 'Delta'], matrix=[[0], [7, 0], [8, 4, 0], [9, 5, 6, 0]])
    >>> del m['Alpha']
    >>> m
    _Matrix(names=['Beta', 'Gamma', 'Delta'], matrix=[[0], [4, 0], [5, 6, 0]])
    >>> m.insert('Alpha', [0, 7, 8, 9] , 0)
    >>> m
    _Matrix(names=['Alpha', 'Beta', 'Gamma', 'Delta'], matrix=[[0], [7, 0], [8, 4, 0], [9, 5, 6, 0]])

    """

    def __init__(self, names, matrix=None):
        """Initialize matrix.

        Arguments are a list of names, and optionally a list of lower
        triangular matrix data (zero matrix used by default).
        """
        if isinstance(names, list) and all((isinstance(s, str) for s in names)):
            if len(set(names)) == len(names):
                self.names = names
            else:
                raise ValueError('Duplicate names found')
        else:
            raise TypeError("'names' should be a list of strings")
        if matrix is None:
            matrix = [[0] * i for i in range(1, len(self) + 1)]
            self.matrix = matrix
        elif isinstance(matrix, list) and all((isinstance(row, list) for row in matrix)) and all((isinstance(item, numbers.Number) for row in matrix for item in row)):
            if len(matrix) == len(names):
                if [len(row) for row in matrix] == list(range(1, len(self) + 1)):
                    self.matrix = matrix
                else:
                    raise ValueError("'matrix' should be in lower triangle format")
            else:
                raise ValueError("'names' and 'matrix' should be the same size")
        else:
            raise TypeError("'matrix' should be a list of numerical lists")

    def __getitem__(self, item):
        """Access value(s) by the index(s) or name(s).

        For a _Matrix object 'dm'::

            dm[i]                   get a value list from the given 'i' to others;
            dm[i, j]                get the value between 'i' and 'j';
            dm['name']              map name to index first
            dm['name1', 'name2']    map name to index first

        """
        if isinstance(item, (int, str)):
            index = None
            if isinstance(item, int):
                index = item
            elif isinstance(item, str):
                if item in self.names:
                    index = self.names.index(item)
                else:
                    raise ValueError('Item not found.')
            else:
                raise TypeError('Invalid index type.')
            if index > len(self) - 1:
                raise IndexError('Index out of range.')
            return [self.matrix[index][i] for i in range(0, index)] + [self.matrix[i][index] for i in range(index, len(self))]
        elif len(item) == 2:
            row_index = None
            col_index = None
            if all((isinstance(i, int) for i in item)):
                row_index, col_index = item
            elif all((isinstance(i, str) for i in item)):
                row_name, col_name = item
                if row_name in self.names and col_name in self.names:
                    row_index = self.names.index(row_name)
                    col_index = self.names.index(col_name)
                else:
                    raise ValueError('Item not found.')
            else:
                raise TypeError('Invalid index type.')
            if row_index > len(self) - 1 or col_index > len(self) - 1:
                raise IndexError('Index out of range.')
            if row_index > col_index:
                return self.matrix[row_index][col_index]
            else:
                return self.matrix[col_index][row_index]
        else:
            raise TypeError('Invalid index type.')

    def __setitem__(self, item, value):
        """Set value by the index(s) or name(s).

        Similar to __getitem__::

            dm[1] = [1, 0, 3, 4]    set values from '1' to others;
            dm[i, j] = 2            set the value from 'i' to 'j'

        """
        if isinstance(item, (int, str)):
            index = None
            if isinstance(item, int):
                index = item
            elif isinstance(item, str):
                if item in self.names:
                    index = self.names.index(item)
                else:
                    raise ValueError('Item not found.')
            else:
                raise TypeError('Invalid index type.')
            if index > len(self) - 1:
                raise IndexError('Index out of range.')
            if isinstance(value, list) and all((isinstance(n, numbers.Number) for n in value)):
                if len(value) == len(self):
                    for i in range(0, index):
                        self.matrix[index][i] = value[i]
                    for i in range(index, len(self)):
                        self.matrix[i][index] = value[i]
                else:
                    raise ValueError('Value not the same size.')
            else:
                raise TypeError('Invalid value type.')
        elif len(item) == 2:
            row_index = None
            col_index = None
            if all((isinstance(i, int) for i in item)):
                row_index, col_index = item
            elif all((isinstance(i, str) for i in item)):
                row_name, col_name = item
                if row_name in self.names and col_name in self.names:
                    row_index = self.names.index(row_name)
                    col_index = self.names.index(col_name)
                else:
                    raise ValueError('Item not found.')
            else:
                raise TypeError('Invalid index type.')
            if row_index > len(self) - 1 or col_index > len(self) - 1:
                raise IndexError('Index out of range.')
            if isinstance(value, numbers.Number):
                if row_index > col_index:
                    self.matrix[row_index][col_index] = value
                else:
                    self.matrix[col_index][row_index] = value
            else:
                raise TypeError('Invalid value type.')
        else:
            raise TypeError('Invalid index type.')

    def __delitem__(self, item):
        """Delete related distances by the index or name."""
        index = None
        if isinstance(item, int):
            index = item
        elif isinstance(item, str):
            index = self.names.index(item)
        else:
            raise TypeError('Invalid index type.')
        for i in range(index + 1, len(self)):
            del self.matrix[i][index]
        del self.matrix[index]
        del self.names[index]

    def insert(self, name, value, index=None):
        """Insert distances given the name and value.

        :Parameters:
            name : str
                name of a row/col to be inserted
            value : list
                a row/col of values to be inserted

        """
        if isinstance(name, str):
            if index is None:
                index = len(self)
            if not isinstance(index, int):
                raise TypeError('Invalid index type.')
            self.names.insert(index, name)
            self.matrix.insert(index, [0] * index)
            for i in range(index, len(self)):
                self.matrix[i].insert(index, 0)
            self[index] = value
        else:
            raise TypeError('Invalid name type.')

    def __len__(self):
        """Matrix length."""
        return len(self.names)

    def __repr__(self):
        """Return Matrix as a string."""
        return self.__class__.__name__ + '(names=%s, matrix=%s)' % tuple(map(repr, (self.names, self.matrix)))

    def __str__(self):
        """Get a lower triangular matrix string."""
        matrix_string = '\n'.join([self.names[i] + '\t' + '\t'.join([format(n, 'f') for n in self.matrix[i]]) for i in range(0, len(self))])
        matrix_string = matrix_string + '\n\t' + '\t'.join(self.names)
        return matrix_string.expandtabs(tabsize=4)