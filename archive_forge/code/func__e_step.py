from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from joblib import effective_n_jobs
from scipy.special import gammaln, logsumexp
from ..base import (
from ..utils import check_random_state, gen_batches, gen_even_slices
from ..utils._param_validation import Interval, StrOptions
from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted, check_non_negative
from ._online_lda_fast import (
from ._online_lda_fast import (
from ._online_lda_fast import (
def _e_step(self, X, cal_sstats, random_init, parallel=None):
    """E-step in EM update.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Document word matrix.

        cal_sstats : bool
            Parameter that indicate whether to calculate sufficient statistics
            or not. Set ``cal_sstats`` to True when we need to run M-step.

        random_init : bool
            Parameter that indicate whether to initialize document topic
            distribution randomly in the E-step. Set it to True in training
            steps.

        parallel : joblib.Parallel, default=None
            Pre-initialized instance of joblib.Parallel.

        Returns
        -------
        (doc_topic_distr, suff_stats) :
            `doc_topic_distr` is unnormalized topic distribution for each
            document. In the literature, this is called `gamma`.
            `suff_stats` is expected sufficient statistics for the M-step.
            When `cal_sstats == False`, it will be None.

        """
    random_state = self.random_state_ if random_init else None
    n_jobs = effective_n_jobs(self.n_jobs)
    if parallel is None:
        parallel = Parallel(n_jobs=n_jobs, verbose=max(0, self.verbose - 1))
    results = parallel((delayed(_update_doc_distribution)(X[idx_slice, :], self.exp_dirichlet_component_, self.doc_topic_prior_, self.max_doc_update_iter, self.mean_change_tol, cal_sstats, random_state) for idx_slice in gen_even_slices(X.shape[0], n_jobs)))
    doc_topics, sstats_list = zip(*results)
    doc_topic_distr = np.vstack(doc_topics)
    if cal_sstats:
        suff_stats = np.zeros(self.components_.shape, dtype=self.components_.dtype)
        for sstats in sstats_list:
            suff_stats += sstats
        suff_stats *= self.exp_dirichlet_component_
    else:
        suff_stats = None
    return (doc_topic_distr, suff_stats)