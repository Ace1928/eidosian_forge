from collections.abc import Callable
from sympy.core.containers import Dict
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int
from .matrices import MatrixBase
from .repmatrix import MutableRepMatrix, RepMatrix
from .utilities import _iszero
from .decompositions import (
from .solvers import (
@classmethod
def _handle_creation_inputs(cls, *args, **kwargs):
    if len(args) == 1 and isinstance(args[0], MatrixBase):
        rows = args[0].rows
        cols = args[0].cols
        smat = args[0].todok()
        return (rows, cols, smat)
    smat = {}
    if len(args) == 2 and args[0] is None:
        args = [None, None, args[1]]
    if len(args) == 3:
        r, c = args[:2]
        if r is c is None:
            rows = cols = None
        elif None in (r, c):
            raise ValueError('Pass rows=None and no cols for autosizing.')
        else:
            rows, cols = (as_int(args[0]), as_int(args[1]))
        if isinstance(args[2], Callable):
            op = args[2]
            if None in (rows, cols):
                raise ValueError('{} and {} must be integers for this specification.'.format(rows, cols))
            row_indices = [cls._sympify(i) for i in range(rows)]
            col_indices = [cls._sympify(j) for j in range(cols)]
            for i in row_indices:
                for j in col_indices:
                    value = cls._sympify(op(i, j))
                    if value != cls.zero:
                        smat[i, j] = value
            return (rows, cols, smat)
        elif isinstance(args[2], (dict, Dict)):

            def update(i, j, v):
                if v:
                    if (i, j) in smat and v != smat[i, j]:
                        raise ValueError('There is a collision at {} for {} and {}.'.format((i, j), v, smat[i, j]))
                    smat[i, j] = v
            for (r, c), v in args[2].items():
                if isinstance(v, MatrixBase):
                    for (i, j), vv in v.todok().items():
                        update(r + i, c + j, vv)
                elif isinstance(v, (list, tuple)):
                    _, _, smat = cls._handle_creation_inputs(v, **kwargs)
                    for i, j in smat:
                        update(r + i, c + j, smat[i, j])
                else:
                    v = cls._sympify(v)
                    update(r, c, cls._sympify(v))
        elif is_sequence(args[2]):
            flat = not any((is_sequence(i) for i in args[2]))
            if not flat:
                _, _, smat = cls._handle_creation_inputs(args[2], **kwargs)
            else:
                flat_list = args[2]
                if len(flat_list) != rows * cols:
                    raise ValueError('The length of the flat list ({}) does not match the specified size ({} * {}).'.format(len(flat_list), rows, cols))
                for i in range(rows):
                    for j in range(cols):
                        value = flat_list[i * cols + j]
                        value = cls._sympify(value)
                        if value != cls.zero:
                            smat[i, j] = value
        if rows is None:
            keys = smat.keys()
            rows = max([r for r, _ in keys]) + 1 if keys else 0
            cols = max([c for _, c in keys]) + 1 if keys else 0
        else:
            for i, j in smat.keys():
                if i and i >= rows or (j and j >= cols):
                    raise ValueError('The location {} is out of the designated range[{}, {}]x[{}, {}]'.format((i, j), 0, rows - 1, 0, cols - 1))
        return (rows, cols, smat)
    elif len(args) == 1 and isinstance(args[0], (list, tuple)):
        v = args[0]
        c = 0
        for i, row in enumerate(v):
            if not isinstance(row, (list, tuple)):
                row = [row]
            for j, vv in enumerate(row):
                if vv != cls.zero:
                    smat[i, j] = cls._sympify(vv)
            c = max(c, len(row))
        rows = len(v) if c else 0
        cols = c
        return (rows, cols, smat)
    else:
        rows, cols, mat = super()._handle_creation_inputs(*args)
        for i in range(rows):
            for j in range(cols):
                value = mat[cols * i + j]
                if value != cls.zero:
                    smat[i, j] = value
        return (rows, cols, smat)