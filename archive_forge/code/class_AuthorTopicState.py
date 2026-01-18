import logging
from itertools import chain
from copy import deepcopy
from shutil import copyfile
from os.path import isfile
from os import remove
import numpy as np  # for arrays, array broadcasting etc.
from scipy.special import gammaln  # gamma function utils
from gensim import utils
from gensim.models import LdaModel
from gensim.models.ldamodel import LdaState
from gensim.matutils import dirichlet_expectation, mean_absolute_difference
from gensim.corpora import MmCorpus
class AuthorTopicState(LdaState):
    """Encapsulate information for computation of :class:`~gensim.models.atmodel.AuthorTopicModel`."""

    def __init__(self, eta, lambda_shape, gamma_shape):
        """

        Parameters
        ----------
        eta: numpy.ndarray
            Dirichlet topic parameter for sparsity.
        lambda_shape: (int, int)
            Initialize topic parameters.
        gamma_shape: int
            Initialize topic parameters.

        """
        self.eta = eta
        self.sstats = np.zeros(lambda_shape)
        self.gamma = np.zeros(gamma_shape)
        self.numdocs = 0
        self.dtype = np.float64