import copy
import numbers
from collections.abc import MutableMapping
from warnings import warn
import numpy as np
from nibabel.affines import apply_affine
from .array_sequence import ArraySequence
class LazyTractogram(Tractogram):
    """Lazy container for streamlines and their data information.

    This container behaves lazily as it uses generator functions to manage
    streamlines and their data information. This container is thus memory
    friendly since it doesn't require having all this data loaded in memory.

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
    streamlines : generator function
        Generator function yielding streamlines. Each streamline is an
        ndarray of shape ($N_t$, 3) where $N_t$ is the number of points of
        streamline $t$.
    data_per_streamline : instance of :class:`LazyDict`
        Dictionary where the items are (str, instantiated generator).
        Each key represents a piece of information $i$ to be kept alongside
        every streamline, and its associated value is a generator function
        yielding that information via ndarrays of shape ($P_i$,) where $P_i$ is
        the number of values to store for that particular piece of information
        $i$.
    data_per_point : :class:`LazyDict` object
        Dictionary where the items are (str, instantiated generator).  Each key
        represents a piece of information $i$ to be kept alongside every point
        of every streamline, and its associated value is a generator function
        yielding that information via ndarrays of shape ($N_t$, $M_i$) where
        $N_t$ is the number of points for a particular streamline $t$ and $M_i$
        is the number of values to store for that particular piece of
        information $i$.

    Notes
    -----
    LazyTractogram objects do not support indexing currently.
    LazyTractogram objects are suited for operations that can be linearized
    such as applying an affine transformation or converting streamlines from
    one file format to another.

    References
    ----------
    .. [#] http://nipy.org/nibabel/coordinate_systems.html#naming-reference-spaces
    .. [#] http://nipy.org/nibabel/coordinate_systems.html#voxel-coordinates-are-in-voxel-space
    """

    def __init__(self, streamlines=None, data_per_streamline=None, data_per_point=None, affine_to_rasmm=None):
        """
        Parameters
        ----------
        streamlines : generator function, optional
            Generator function yielding streamlines. Each streamline is an
            ndarray of shape ($N_t$, 3) where $N_t$ is the number of points of
            streamline $t$.
        data_per_streamline : dict of generator functions, optional
            Dictionary where the items are (str, generator function).
            Each key represents an information $i$ to be kept alongside every
            streamline, and its associated value is a generator function
            yielding that information via ndarrays of shape ($P_i$,) where
            $P_i$ is the number of values to store for that particular
            information $i$.
        data_per_point : dict of generator functions, optional
            Dictionary where the items are (str, generator function).
            Each key represents an information $i$ to be kept alongside every
            point of every streamline, and its associated value is a generator
            function yielding that information via ndarrays of shape
            ($N_t$, $M_i$) where $N_t$ is the number of points for a particular
            streamline $t$ and $M_i$ is the number of values to store for
            that particular information $i$.
        affine_to_rasmm : ndarray of shape (4, 4) or None, optional
            Transformation matrix that brings the streamlines contained in
            this tractogram to *RAS+* and *mm* space where coordinate (0,0,0)
            refers to the center of the voxel. By default, the streamlines
            are in an unknown space, i.e. affine_to_rasmm is None.
        """
        super().__init__(streamlines, data_per_streamline, data_per_point, affine_to_rasmm)
        self._nb_streamlines = None
        self._data = None
        self._affine_to_apply = np.eye(4)

    @classmethod
    def from_tractogram(cls, tractogram):
        """Creates a :class:`LazyTractogram` object from a :class:`Tractogram` object.

        Parameters
        ----------
        tractogram : :class:`Tractgogram` object
            Tractogram from which to create a :class:`LazyTractogram` object.

        Returns
        -------
        lazy_tractogram : :class:`LazyTractogram` object
            New lazy tractogram.
        """
        lazy_tractogram = cls(lambda: tractogram.streamlines.copy())

        def _gen(key):
            return lambda: iter(tractogram.data_per_streamline[key])
        for k in tractogram.data_per_streamline:
            lazy_tractogram._data_per_streamline[k] = _gen(k)

        def _gen(key):
            return lambda: iter(tractogram.data_per_point[key])
        for k in tractogram.data_per_point:
            lazy_tractogram._data_per_point[k] = _gen(k)
        lazy_tractogram._nb_streamlines = len(tractogram)
        lazy_tractogram.affine_to_rasmm = tractogram.affine_to_rasmm
        return lazy_tractogram

    @classmethod
    def from_data_func(cls, data_func):
        """Creates an instance from a generator function.

        The generator function must yield :class:`TractogramItem` objects.

        Parameters
        ----------
        data_func : generator function yielding :class:`TractogramItem` objects
            Generator function that whenever is called starts yielding
            :class:`TractogramItem` objects that will be used to instantiate a
            :class:`LazyTractogram`.

        Returns
        -------
        lazy_tractogram : :class:`LazyTractogram` object
            New lazy tractogram.
        """
        if not callable(data_func):
            raise TypeError('`data_func` must be a generator function.')
        lazy_tractogram = cls()
        lazy_tractogram._data = data_func
        try:
            first_item = next(data_func())

            def _gen(key):
                return lambda: (t.data_for_streamline[key] for t in data_func())
            data_per_streamline_keys = first_item.data_for_streamline.keys()
            for k in data_per_streamline_keys:
                lazy_tractogram._data_per_streamline[k] = _gen(k)

            def _gen(key):
                return lambda: (t.data_for_points[key] for t in data_func())
            data_per_point_keys = first_item.data_for_points.keys()
            for k in data_per_point_keys:
                lazy_tractogram._data_per_point[k] = _gen(k)
        except StopIteration:
            pass
        return lazy_tractogram

    @property
    def streamlines(self):
        streamlines_gen = iter([])
        if self._streamlines is not None:
            streamlines_gen = self._streamlines()
        elif self._data is not None:
            streamlines_gen = (t.streamline for t in self._data())
        if not np.allclose(self._affine_to_apply, np.eye(4)):

            def _apply_affine():
                for s in streamlines_gen:
                    yield apply_affine(self._affine_to_apply, s)
            return _apply_affine()
        return streamlines_gen

    def _set_streamlines(self, value):
        if value is not None and (not callable(value)):
            msg = '`streamlines` must be a generator function. That is a function which, when called, returns an instantiated generator.'
            raise TypeError(msg)
        self._streamlines = value

    @property
    def data_per_streamline(self):
        return self._data_per_streamline

    @data_per_streamline.setter
    def data_per_streamline(self, value):
        self._data_per_streamline = LazyDict(value)

    @property
    def data_per_point(self):
        return self._data_per_point

    @data_per_point.setter
    def data_per_point(self, value):
        self._data_per_point = LazyDict(value)

    @property
    def data(self):
        if self._data is not None:
            return self._data()

        def _gen_data():
            data_per_streamline_generators = {}
            for k, v in self.data_per_streamline.items():
                data_per_streamline_generators[k] = iter(v)
            data_per_point_generators = {}
            for k, v in self.data_per_point.items():
                data_per_point_generators[k] = iter(v)
            for s in self.streamlines:
                data_for_streamline = {}
                for k, v in data_per_streamline_generators.items():
                    data_for_streamline[k] = next(v)
                data_for_points = {}
                for k, v in data_per_point_generators.items():
                    data_for_points[k] = next(v)
                yield TractogramItem(s, data_for_streamline, data_for_points)
        return _gen_data()

    def __getitem__(self, idx):
        raise NotImplementedError('LazyTractogram does not support indexing.')

    def extend(self, other):
        msg = 'LazyTractogram does not support concatenation.'
        raise NotImplementedError(msg)

    def __iter__(self):
        count = 0
        for tractogram_item in self.data:
            yield tractogram_item
            count += 1
        self._nb_streamlines = count

    def __len__(self):
        if self._nb_streamlines is None:
            warn('Number of streamlines will be determined manually by looping through the streamlines. If you know the actual number of streamlines, you might want to set it beforehand via `self.header.nb_streamlines`.', Warning)
            self._nb_streamlines = sum((1 for _ in self.streamlines))
        return self._nb_streamlines

    def copy(self):
        """Returns a copy of this :class:`LazyTractogram` object."""
        tractogram = LazyTractogram(self._streamlines, self._data_per_streamline, self._data_per_point, self.affine_to_rasmm)
        tractogram._nb_streamlines = self._nb_streamlines
        tractogram._data = self._data
        tractogram._affine_to_apply = self._affine_to_apply.copy()
        return tractogram

    def apply_affine(self, affine, lazy=True):
        """Applies an affine transformation to the streamlines.

        The transformation given by the `affine` matrix is applied after any
        other pending transformations to the streamline points.

        Parameters
        ----------
        affine : 2D array (4,4)
            Transformation matrix that will be applied on each streamline.
        lazy : True, optional
            Should always be True for :class:`LazyTractogram` object. Doing
            otherwise will raise a ValueError.

        Returns
        -------
        lazy_tractogram : :class:`LazyTractogram` object
            A copy of this :class:`LazyTractogram` instance but with a
            transformation to be applied on the streamlines.
        """
        if not lazy:
            msg = 'LazyTractogram only supports lazy transformations.'
            raise ValueError(msg)
        tractogram = self.copy()
        tractogram._affine_to_apply = np.dot(affine, self._affine_to_apply)
        if tractogram.affine_to_rasmm is not None:
            tractogram.affine_to_rasmm = np.dot(self.affine_to_rasmm, np.linalg.inv(affine))
        return tractogram

    def to_world(self, lazy=True):
        """Brings the streamlines to world space (i.e. RAS+ and mm).

        The transformation is applied after any other pending transformations
        to the streamline points.

        Parameters
        ----------
        lazy : True, optional
            Should always be True for :class:`LazyTractogram` object. Doing
            otherwise will raise a ValueError.

        Returns
        -------
        lazy_tractogram : :class:`LazyTractogram` object
            A copy of this :class:`LazyTractogram` instance but with a
            transformation to be applied on the streamlines.
        """
        if self.affine_to_rasmm is None:
            msg = "Streamlines are in a unknown space. This error can be avoided by setting the 'affine_to_rasmm' property."
            raise ValueError(msg)
        return self.apply_affine(self.affine_to_rasmm, lazy=lazy)