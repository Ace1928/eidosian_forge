import copy
from abc import abstractmethod
from math import sqrt
from sys import stdout
from nltk.cluster.api import ClusterI
class VectorSpaceClusterer(ClusterI):
    """
    Abstract clusterer which takes tokens and maps them into a vector space.
    Optionally performs singular value decomposition to reduce the
    dimensionality.
    """

    def __init__(self, normalise=False, svd_dimensions=None):
        """
        :param normalise:       should vectors be normalised to length 1
        :type normalise:        boolean
        :param svd_dimensions:  number of dimensions to use in reducing vector
                                dimensionsionality with SVD
        :type svd_dimensions:   int
        """
        self._Tt = None
        self._should_normalise = normalise
        self._svd_dimensions = svd_dimensions

    def cluster(self, vectors, assign_clusters=False, trace=False):
        assert len(vectors) > 0
        if self._should_normalise:
            vectors = list(map(self._normalise, vectors))
        if self._svd_dimensions and self._svd_dimensions < len(vectors[0]):
            [u, d, vt] = numpy.linalg.svd(numpy.transpose(numpy.array(vectors)))
            S = d[:self._svd_dimensions] * numpy.identity(self._svd_dimensions, numpy.float64)
            T = u[:, :self._svd_dimensions]
            Dt = vt[:self._svd_dimensions, :]
            vectors = numpy.transpose(numpy.dot(S, Dt))
            self._Tt = numpy.transpose(T)
        self.cluster_vectorspace(vectors, trace)
        if assign_clusters:
            return [self.classify(vector) for vector in vectors]

    @abstractmethod
    def cluster_vectorspace(self, vectors, trace):
        """
        Finds the clusters using the given set of vectors.
        """

    def classify(self, vector):
        if self._should_normalise:
            vector = self._normalise(vector)
        if self._Tt is not None:
            vector = numpy.dot(self._Tt, vector)
        cluster = self.classify_vectorspace(vector)
        return self.cluster_name(cluster)

    @abstractmethod
    def classify_vectorspace(self, vector):
        """
        Returns the index of the appropriate cluster for the vector.
        """

    def likelihood(self, vector, label):
        if self._should_normalise:
            vector = self._normalise(vector)
        if self._Tt is not None:
            vector = numpy.dot(self._Tt, vector)
        return self.likelihood_vectorspace(vector, label)

    def likelihood_vectorspace(self, vector, cluster):
        """
        Returns the likelihood of the vector belonging to the cluster.
        """
        predicted = self.classify_vectorspace(vector)
        return 1.0 if cluster == predicted else 0.0

    def vector(self, vector):
        """
        Returns the vector after normalisation and dimensionality reduction
        """
        if self._should_normalise:
            vector = self._normalise(vector)
        if self._Tt is not None:
            vector = numpy.dot(self._Tt, vector)
        return vector

    def _normalise(self, vector):
        """
        Normalises the vector to unit length.
        """
        return vector / sqrt(numpy.dot(vector, vector))