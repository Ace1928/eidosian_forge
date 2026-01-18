from __future__ import division
from . import matrix
from . import utils
from .base import DataGraph
from .base import PyGSPGraph
from builtins import super
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd
import numbers
import numpy as np
import tasklogger
import warnings
class LandmarkGraph(DataGraph):
    """Landmark graph

    Adds landmarking feature to any data graph by taking spectral clusters
    and building a 'landmark operator' from clusters to samples and back to
    clusters.
    Any transformation on the landmark kernel is trivially extended to the
    data space by multiplying by the transition matrix.

    Parameters
    ----------

    data : array-like, shape=[n_samples,n_features]
        accepted types: `numpy.ndarray`, `scipy.sparse.spmatrix`.,
        `pandas.DataFrame`, `pandas.SparseDataFrame`.

    n_landmark : `int`, optional (default: 2000)
        number of landmarks to use

    n_svd : `int`, optional (default: 100)
        number of SVD components to use for spectral clustering

    Attributes
    ----------
    landmark_op : array-like, shape=[n_landmark, n_landmark]
        Landmark operator.
        Can be treated as a diffusion operator between landmarks.

    transitions : array-like, shape=[n_samples, n_landmark]
        Transition probabilities between samples and landmarks.

    clusters : array-like, shape=[n_samples]
        Private attribute. Cluster assignments for each sample.

    Examples
    --------
    >>> G = graphtools.Graph(data, n_landmark=1000)
    >>> X_landmark = transform(G.landmark_op)
    >>> X_full = G.interpolate(X_landmark)
    """

    def __init__(self, data, n_landmark=2000, n_svd=100, **kwargs):
        """Initialize a landmark graph.

        Raises
        ------
        RuntimeWarning : if too many SVD dimensions or
        too few landmarks are used
        """
        if n_landmark >= data.shape[0]:
            raise ValueError('n_landmark ({}) >= n_samples ({}). Use kNNGraph instead'.format(n_landmark, data.shape[0]))
        if n_svd >= data.shape[0]:
            warnings.warn('n_svd ({}) >= n_samples ({}) Consider using kNNGraph or lower n_svd'.format(n_svd, data.shape[0]), RuntimeWarning)
        self.n_landmark = n_landmark
        self.n_svd = n_svd
        super().__init__(data, **kwargs)

    def get_params(self):
        """Get parameters from this object"""
        params = super().get_params()
        params.update({'n_landmark': self.n_landmark, 'n_pca': self.n_pca})
        return params

    def set_params(self, **params):
        """Set parameters on this object

        Safe setter method - attributes should not be modified directly as some
        changes are not valid.
        Valid parameters:
        - n_landmark
        - n_svd

        Parameters
        ----------
        params : key-value pairs of parameter name and new values

        Returns
        -------
        self
        """
        reset_landmarks = False
        if 'n_landmark' in params and params['n_landmark'] != self.n_landmark:
            self.n_landmark = params['n_landmark']
            reset_landmarks = True
        if 'n_svd' in params and params['n_svd'] != self.n_svd:
            self.n_svd = params['n_svd']
            reset_landmarks = True
        super().set_params(**params)
        if reset_landmarks:
            self._reset_landmarks()
        return self

    def _reset_landmarks(self):
        """Reset landmark data

        Landmarks can be recomputed without recomputing the kernel
        """
        try:
            del self._landmark_op
            del self._transitions
            del self._clusters
        except AttributeError:
            pass

    @property
    def landmark_op(self):
        """Landmark operator

        Compute or return the landmark operator

        Returns
        -------
        landmark_op : array-like, shape=[n_landmark, n_landmark]
            Landmark operator. Can be treated as a diffusion operator between
            landmarks.
        """
        try:
            return self._landmark_op
        except AttributeError:
            self.build_landmark_op()
            return self._landmark_op

    @property
    def clusters(self):
        """Cluster assignments for each sample.

        Compute or return the cluster assignments

        Returns
        -------
        clusters : list-like, shape=[n_samples]
            Cluster assignments for each sample.
        """
        try:
            return self._clusters
        except AttributeError:
            self.build_landmark_op()
            return self._clusters

    @property
    def transitions(self):
        """Transition matrix from samples to landmarks

        Compute the landmark operator if necessary, then return the
        transition matrix.

        Returns
        -------
        transitions : array-like, shape=[n_samples, n_landmark]
            Transition probabilities between samples and landmarks.
        """
        try:
            return self._transitions
        except AttributeError:
            self.build_landmark_op()
            return self._transitions

    def _landmarks_to_data(self):
        landmarks = np.unique(self.clusters)
        if sparse.issparse(self.kernel):
            pmn = sparse.vstack([sparse.csr_matrix(self.kernel[self.clusters == i, :].sum(axis=0)) for i in landmarks])
        else:
            pmn = np.array([np.sum(self.kernel[self.clusters == i, :], axis=0) for i in landmarks])
        return pmn

    def _data_transitions(self):
        return normalize(self._landmarks_to_data(), 'l1', axis=1)

    def build_landmark_op(self):
        """Build the landmark operator

        Calculates spectral clusters on the kernel, and calculates transition
        probabilities between cluster centers by using transition probabilities
        between samples assigned to each cluster.
        """
        with _logger.log_task('landmark operator'):
            is_sparse = sparse.issparse(self.kernel)
            with _logger.log_task('SVD'):
                _, _, VT = randomized_svd(self.diff_aff, n_components=self.n_svd, random_state=self.random_state)
            with _logger.log_task('KMeans'):
                kmeans = MiniBatchKMeans(self.n_landmark, init_size=3 * self.n_landmark, n_init=1, batch_size=10000, random_state=self.random_state)
                self._clusters = kmeans.fit_predict(self.diff_op.dot(VT.T))
            pmn = self._landmarks_to_data()
            pnm = pmn.transpose()
            pmn = normalize(pmn, norm='l1', axis=1)
            pnm = normalize(pnm, norm='l1', axis=1)
            landmark_op = pmn.dot(pnm)
            if is_sparse:
                landmark_op = landmark_op.toarray()
            self._landmark_op = landmark_op
            self._transitions = pnm

    def extend_to_data(self, data, **kwargs):
        """Build transition matrix from new data to the graph

        Creates a transition matrix such that `Y` can be approximated by
        a linear combination of landmarks. Any
        transformation of the landmarks can be trivially applied to `Y` by
        performing

        `transform_Y = transitions.dot(transform)`

        Parameters
        ----------

        Y: array-like, [n_samples_y, n_features]
            new data for which an affinity matrix is calculated
            to the existing data. `n_features` must match
            either the ambient or PCA dimensions

        Returns
        -------

        transitions : array-like, [n_samples_y, self.data.shape[0]]
            Transition matrix from `Y` to `self.data`
        """
        kernel = self.build_kernel_to_data(data, **kwargs)
        if sparse.issparse(kernel):
            pnm = sparse.hstack([sparse.csr_matrix(kernel[:, self.clusters == i].sum(axis=1)) for i in np.unique(self.clusters)])
        else:
            pnm = np.array([np.sum(kernel[:, self.clusters == i], axis=1).T for i in np.unique(self.clusters)]).transpose()
        pnm = normalize(pnm, norm='l1', axis=1)
        return pnm

    def interpolate(self, transform, transitions=None, Y=None):
        """Interpolate new data onto a transformation of the graph data

        One of either transitions or Y should be provided

        Parameters
        ----------

        transform : array-like, shape=[n_samples, n_transform_features]

        transitions : array-like, optional, shape=[n_samples_y, n_samples]
            Transition matrix from `Y` (not provided) to `self.data`

        Y: array-like, optional, shape=[n_samples_y, n_features]
            new data for which an affinity matrix is calculated
            to the existing data. `n_features` must match
            either the ambient or PCA dimensions

        Returns
        -------

        Y_transform : array-like, [n_samples_y, n_features or n_pca]
            Transition matrix from `Y` to `self.data`
        """
        if transitions is None and Y is None:
            transitions = self.transitions
        return super().interpolate(transform, transitions=transitions, Y=Y)