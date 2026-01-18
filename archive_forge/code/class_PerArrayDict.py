import copy
import numbers
from collections.abc import MutableMapping
from warnings import warn
import numpy as np
from nibabel.affines import apply_affine
from .array_sequence import ArraySequence
class PerArrayDict(SliceableDataDict):
    """Dictionary for which key access can do slicing on the values.

    This container behaves like a standard dictionary but extends key access to
    allow keys for key access to be indices slicing into the contained ndarray
    values. The elements must also be ndarrays.

    In addition, it makes sure the amount of data contained in those ndarrays
    matches the number of streamlines given at the instantiation of this
    instance.

    Parameters
    ----------
    n_rows : None or int, optional
        Number of rows per value in each key, value pair or None for not
        specified.
    \\*args :
    \\*\\*kwargs :
        Positional and keyword arguments, passed straight through the ``dict``
        constructor.
    """

    def __init__(self, n_rows=0, *args, **kwargs):
        self.n_rows = n_rows
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        value = np.asarray(list(value))
        if value.ndim == 1 and value.dtype != object:
            value.shape = (len(value), 1)
        if value.ndim != 2:
            raise ValueError('data_per_streamline must be a 2D array.')
        if 0 < self.n_rows != len(value):
            msg = f'The number of values ({len(value)}) should match n_elements ({self.n_rows}).'
            raise ValueError(msg)
        self.store[key] = value

    def _extend_entry(self, key, value):
        """Appends the `value` to the entry specified by `key`."""
        self[key] = np.concatenate([self[key], value])

    def extend(self, other):
        """Appends the elements of another :class:`PerArrayDict`.

        That is, for each entry in this dictionary, we append the elements
        coming from the other dictionary at the corresponding entry.

        Parameters
        ----------
        other : :class:`PerArrayDict` object
            Its data will be appended to the data of this dictionary.

        Returns
        -------
        None

        Notes
        -----
        The keys in both dictionaries must be the same.
        """
        if len(self) > 0 and len(other) > 0 and (sorted(self.keys()) != sorted(other.keys())):
            msg = f"Entry mismatched between the two PerArrayDict objects. This PerArrayDict contains '{sorted(self.keys())}' whereas the other contains '{sorted(other.keys())}'."
            raise ValueError(msg)
        self.n_rows += other.n_rows
        for key in other.keys():
            if key not in self:
                self[key] = other[key]
            else:
                self._extend_entry(key, other[key])