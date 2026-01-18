import copy
import numbers
from collections.abc import MutableMapping
from warnings import warn
import numpy as np
from nibabel.affines import apply_affine
from .array_sequence import ArraySequence
class Tractogram:
    """Container for streamlines and their data information.

    Streamlines of a tractogram can be in any coordinate system of your
    choice as long as you provide the correct `affine_to_rasmm` matrix, at
    construction time. When applied to streamlines coordinates, that
    transformation matrix should bring the streamlines back to world space
    (RAS+ and mm space) [#]_.

    Moreover, when streamlines are mapped back to voxel space [#]_, a
    streamline point located at an integer coordinate (i,j,k) is considered
    to be at the center of the corresponding voxel. This is in contrast with
    other conventions where it might have referred to a corner.

    Attributes
    ----------
    streamlines : :class:`ArraySequence` object
        Sequence of $T$ streamlines. Each streamline is an ndarray of
        shape ($N_t$, 3) where $N_t$ is the number of points of
        streamline $t$.
    data_per_streamline : :class:`PerArrayDict` object
        Dictionary where the items are (str, 2D array).  Each key represents a
        piece of information $i$ to be kept alongside every streamline, and its
        associated value is a 2D array of shape ($T$, $P_i$) where $T$ is the
        number of streamlines and $P_i$ is the number of values to store for
        that particular piece of information $i$.
    data_per_point : :class:`PerArraySequenceDict` object
        Dictionary where the items are (str, :class:`ArraySequence`).  Each key
        represents a piece of information $i$ to be kept alongside every point
        of every streamline, and its associated value is an iterable of
        ndarrays of shape ($N_t$, $M_i$) where $N_t$ is the number of points
        for a particular streamline $t$ and $M_i$ is the number values to store
        for that particular piece of information $i$.

    References
    ----------
    .. [#] http://nipy.org/nibabel/coordinate_systems.html#naming-reference-spaces
    .. [#] http://nipy.org/nibabel/coordinate_systems.html#voxel-coordinates-are-in-voxel-space
    """

    def __init__(self, streamlines=None, data_per_streamline=None, data_per_point=None, affine_to_rasmm=None):
        """
        Parameters
        ----------
        streamlines : iterable of ndarrays or :class:`ArraySequence`, optional
            Sequence of $T$ streamlines. Each streamline is an ndarray of
            shape ($N_t$, 3) where $N_t$ is the number of points of
            streamline $t$.
        data_per_streamline : dict of iterable of ndarrays, optional
            Dictionary where the items are (str, iterable).
            Each key represents an information $i$ to be kept alongside every
            streamline, and its associated value is an iterable of ndarrays of
            shape ($P_i$,) where $P_i$ is the number of scalar values to store
            for that particular information $i$.
        data_per_point : dict of iterable of ndarrays, optional
            Dictionary where the items are (str, iterable).
            Each key represents an information $i$ to be kept alongside every
            point of every streamline, and its associated value is an iterable
            of ndarrays of shape ($N_t$, $M_i$) where $N_t$ is the number of
            points for a particular streamline $t$ and $M_i$ is the number
            scalar values to store for that particular information $i$.
        affine_to_rasmm : ndarray of shape (4, 4) or None, optional
            Transformation matrix that brings the streamlines contained in
            this tractogram to *RAS+* and *mm* space where coordinate (0,0,0)
            refers to the center of the voxel. By default, the streamlines
            are in an unknown space, i.e. affine_to_rasmm is None.
        """
        self._set_streamlines(streamlines)
        self.data_per_streamline = data_per_streamline
        self.data_per_point = data_per_point
        self.affine_to_rasmm = affine_to_rasmm

    @property
    def streamlines(self):
        return self._streamlines

    def _set_streamlines(self, value):
        self._streamlines = ArraySequence(value)

    @property
    def data_per_streamline(self):
        return self._data_per_streamline

    @data_per_streamline.setter
    def data_per_streamline(self, value):
        self._data_per_streamline = PerArrayDict(len(self.streamlines), {} if value is None else value)

    @property
    def data_per_point(self):
        return self._data_per_point

    @data_per_point.setter
    def data_per_point(self, value):
        self._data_per_point = PerArraySequenceDict(self.streamlines.total_nb_rows, {} if value is None else value)

    @property
    def affine_to_rasmm(self):
        """Affine bringing streamlines in this tractogram to RAS+mm."""
        return copy.deepcopy(self._affine_to_rasmm)

    @affine_to_rasmm.setter
    def affine_to_rasmm(self, value):
        if value is not None:
            value = np.array(value)
            if value.shape != (4, 4):
                msg = f'Affine matrix has a shape of (4, 4) but a ndarray with shape {value.shape} was provided instead.'
                raise ValueError(msg)
        self._affine_to_rasmm = value

    def __iter__(self):
        for i in range(len(self.streamlines)):
            yield self[i]

    def __getitem__(self, idx):
        pts = self.streamlines[idx]
        data_per_streamline = {}
        for key in self.data_per_streamline:
            data_per_streamline[key] = self.data_per_streamline[key][idx]
        data_per_point = {}
        for key in self.data_per_point:
            data_per_point[key] = self.data_per_point[key][idx]
        if isinstance(idx, (numbers.Integral, np.integer)):
            return TractogramItem(pts, data_per_streamline, data_per_point)
        return Tractogram(pts, data_per_streamline, data_per_point, affine_to_rasmm=self.affine_to_rasmm)

    def __len__(self):
        return len(self.streamlines)

    def copy(self):
        """Returns a copy of this :class:`Tractogram` object."""
        return copy.deepcopy(self)

    def apply_affine(self, affine, lazy=False):
        """Applies an affine transformation on the points of each streamline.

        If `lazy` is not specified, this is performed *in-place*.

        Parameters
        ----------
        affine : ndarray of shape (4, 4)
            Transformation that will be applied to every streamline.
        lazy : {False, True}, optional
            If True, streamlines are *not* transformed in-place and a
            :class:`LazyTractogram` object is returned. Otherwise, streamlines
            are modified in-place.

        Returns
        -------
        tractogram : :class:`Tractogram` or :class:`LazyTractogram` object
            Tractogram where the streamlines have been transformed according
            to the given affine transformation. If the `lazy` option is true,
            it returns a :class:`LazyTractogram` object, otherwise it returns a
            reference to this :class:`Tractogram` object with updated
            streamlines.
        """
        if lazy:
            lazy_tractogram = LazyTractogram.from_tractogram(self)
            return lazy_tractogram.apply_affine(affine)
        if len(self.streamlines) == 0:
            return self
        if np.all(affine == np.eye(4)):
            return self
        if self.streamlines.is_sliced_view:
            for i in range(len(self.streamlines)):
                self.streamlines[i] = apply_affine(affine, self.streamlines[i])
        else:
            self.streamlines._data = apply_affine(affine, self.streamlines._data, inplace=True)
        if self.affine_to_rasmm is not None:
            self.affine_to_rasmm = np.dot(self.affine_to_rasmm, np.linalg.inv(affine))
        return self

    def to_world(self, lazy=False):
        """Brings the streamlines to world space (i.e. RAS+ and mm).

        If `lazy` is not specified, this is performed *in-place*.

        Parameters
        ----------
        lazy : {False, True}, optional
            If True, streamlines are *not* transformed in-place and a
            :class:`LazyTractogram` object is returned. Otherwise, streamlines
            are modified in-place.

        Returns
        -------
        tractogram : :class:`Tractogram` or :class:`LazyTractogram` object
            Tractogram where the streamlines have been sent to world space.
            If the `lazy` option is true, it returns a :class:`LazyTractogram`
            object, otherwise it returns a reference to this
            :class:`Tractogram` object with updated streamlines.
        """
        if self.affine_to_rasmm is None:
            msg = "Streamlines are in a unknown space. This error can be avoided by setting the 'affine_to_rasmm' property."
            raise ValueError(msg)
        return self.apply_affine(self.affine_to_rasmm, lazy=lazy)

    def extend(self, other):
        """Appends the data of another :class:`Tractogram`.

        Data that will be appended includes the streamlines and the content
        of both dictionaries `data_per_streamline` and `data_per_point`.

        Parameters
        ----------
        other : :class:`Tractogram` object
            Its data will be appended to the data of this tractogram.

        Returns
        -------
        None

        Notes
        -----
        The entries in both dictionaries `self.data_per_streamline` and
        `self.data_per_point` must match respectively those contained in
        the other tractogram.
        """
        self.streamlines.extend(other.streamlines)
        self.data_per_streamline.extend(other.data_per_streamline)
        self.data_per_point.extend(other.data_per_point)

    def __iadd__(self, other):
        self.extend(other)
        return self

    def __add__(self, other):
        tractogram = self.copy()
        tractogram += other
        return tractogram