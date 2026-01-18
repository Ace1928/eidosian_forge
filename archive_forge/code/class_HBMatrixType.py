import warnings
import numpy as np
from scipy.sparse import csc_matrix
from ._fortran_format_parser import FortranFormatParser, IntFormat, ExpFormat
class HBMatrixType:
    """Class to hold the matrix type."""
    _q2f_type = {'real': 'R', 'complex': 'C', 'pattern': 'P', 'integer': 'I'}
    _q2f_structure = {'symmetric': 'S', 'unsymmetric': 'U', 'hermitian': 'H', 'skewsymmetric': 'Z', 'rectangular': 'R'}
    _q2f_storage = {'assembled': 'A', 'elemental': 'E'}
    _f2q_type = {j: i for i, j in _q2f_type.items()}
    _f2q_structure = {j: i for i, j in _q2f_structure.items()}
    _f2q_storage = {j: i for i, j in _q2f_storage.items()}

    @classmethod
    def from_fortran(cls, fmt):
        if not len(fmt) == 3:
            raise ValueError('Fortran format for matrix type should be 3 characters long')
        try:
            value_type = cls._f2q_type[fmt[0]]
            structure = cls._f2q_structure[fmt[1]]
            storage = cls._f2q_storage[fmt[2]]
            return cls(value_type, structure, storage)
        except KeyError as e:
            raise ValueError('Unrecognized format %s' % fmt) from e

    def __init__(self, value_type, structure, storage='assembled'):
        self.value_type = value_type
        self.structure = structure
        self.storage = storage
        if value_type not in self._q2f_type:
            raise ValueError('Unrecognized type %s' % value_type)
        if structure not in self._q2f_structure:
            raise ValueError('Unrecognized structure %s' % structure)
        if storage not in self._q2f_storage:
            raise ValueError('Unrecognized storage %s' % storage)

    @property
    def fortran_format(self):
        return self._q2f_type[self.value_type] + self._q2f_structure[self.structure] + self._q2f_storage[self.storage]

    def __repr__(self):
        return f'HBMatrixType({self.value_type}, {self.structure}, {self.storage})'