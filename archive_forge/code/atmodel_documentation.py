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
Get topic distribution for input `author_names`.

        Parameters
        ----------
        author_names : {str, list of str}
            Name(s) of the author for which the topic distribution needs to be estimated.
        eps : float, optional
            The minimum probability value for showing the topics of a given author, topics with probability < `eps`
            will be ignored.

        Returns
        -------
        list of (int, float) **or** list of list of (int, float)
            Topic distribution for the author(s), type depends on type of `author_names`.

        