import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
class sslm(utils.SaveLoad):
    """Encapsulate the inner State Space Language Model for DTM.

    Some important attributes of this class:

        * `obs` is a matrix containing the document to topic ratios.
        * `e_log_prob` is a matrix containing the topic to word ratios.
        * `mean` contains the mean values to be used for inference for each word for a time slice.
        * `variance` contains the variance values to be used for inference of word in a time slice.
        * `fwd_mean` and`fwd_variance` are the forward posterior values for the mean and the variance.
        * `zeta` is an extra variational parameter with a value for each time slice.

    """

    def __init__(self, vocab_len=None, num_time_slices=None, num_topics=None, obs_variance=0.5, chain_variance=0.005):
        self.vocab_len = vocab_len
        self.num_time_slices = num_time_slices
        self.obs_variance = obs_variance
        self.chain_variance = chain_variance
        self.num_topics = num_topics
        self.obs = np.zeros((vocab_len, num_time_slices))
        self.e_log_prob = np.zeros((vocab_len, num_time_slices))
        self.mean = np.zeros((vocab_len, num_time_slices + 1))
        self.fwd_mean = np.zeros((vocab_len, num_time_slices + 1))
        self.fwd_variance = np.zeros((vocab_len, num_time_slices + 1))
        self.variance = np.zeros((vocab_len, num_time_slices + 1))
        self.zeta = np.zeros(num_time_slices)
        self.m_update_coeff = None
        self.mean_t = None
        self.variance_t = None
        self.influence_sum_lgl = None
        self.w_phi_l = None
        self.w_phi_sum = None
        self.w_phi_l_sq = None
        self.m_update_coeff_g = None

    def update_zeta(self):
        """Update the Zeta variational parameter.

        Zeta is described in the appendix and is equal to sum (exp(mean[word] + Variance[word] / 2)),
        over every time-slice. It is the value of variational parameter zeta which maximizes the lower bound.

        Returns
        -------
        list of float
            The updated zeta values for each time slice.

        """
        for j, val in enumerate(self.zeta):
            self.zeta[j] = np.sum(np.exp(self.mean[:, j + 1] + self.variance[:, j + 1] / 2))
        return self.zeta

    def compute_post_variance(self, word, chain_variance):
        """Get the variance, based on the
        `Variational Kalman Filtering approach for Approximate Inference (section 3.1)
        <https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf>`_.

        This function accepts the word to compute variance for, along with the associated sslm class object,
        and returns the `variance` and the posterior approximation `fwd_variance`.

        Notes
        -----
        This function essentially computes Var[\\beta_{t,w}] for t = 1:T

        .. :math::

            fwd\\_variance[t] \\equiv E((beta_{t,w}-mean_{t,w})^2 |beta_{t}\\ for\\ 1:t) =
            (obs\\_variance / fwd\\_variance[t - 1] + chain\\_variance + obs\\_variance ) *
            (fwd\\_variance[t - 1] + obs\\_variance)

        .. :math::

            variance[t] \\equiv E((beta_{t,w}-mean\\_cap_{t,w})^2 |beta\\_cap_{t}\\ for\\ 1:t) =
            fwd\\_variance[t - 1] + (fwd\\_variance[t - 1] / fwd\\_variance[t - 1] + obs\\_variance)^2 *
            (variance[t - 1] - (fwd\\_variance[t-1] + obs\\_variance))

        Parameters
        ----------
        word: int
            The word's ID.
        chain_variance : float
            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            The first returned value is the variance of each word in each time slice, the second value is the
            inferred posterior variance for the same pairs.

        """
        INIT_VARIANCE_CONST = 1000
        T = self.num_time_slices
        variance = self.variance[word]
        fwd_variance = self.fwd_variance[word]
        fwd_variance[0] = chain_variance * INIT_VARIANCE_CONST
        for t in range(1, T + 1):
            if self.obs_variance:
                c = self.obs_variance / (fwd_variance[t - 1] + chain_variance + self.obs_variance)
            else:
                c = 0
            fwd_variance[t] = c * (fwd_variance[t - 1] + chain_variance)
        variance[T] = fwd_variance[T]
        for t in range(T - 1, -1, -1):
            if fwd_variance[t] > 0.0:
                c = np.power(fwd_variance[t] / (fwd_variance[t] + chain_variance), 2)
            else:
                c = 0
            variance[t] = c * (variance[t + 1] - chain_variance) + (1 - c) * fwd_variance[t]
        return (variance, fwd_variance)

    def compute_post_mean(self, word, chain_variance):
        """Get the mean, based on the `Variational Kalman Filtering approach for Approximate Inference (section 3.1)
        <https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf>`_.

        Notes
        -----
        This function essentially computes E[\x08eta_{t,w}] for t = 1:T.

        .. :math::

            Fwd_Mean(t) ≡  E(beta_{t,w} | beta_ˆ 1:t )
            = (obs_variance / fwd_variance[t - 1] + chain_variance + obs_variance ) * fwd_mean[t - 1] +
            (1 - (obs_variance / fwd_variance[t - 1] + chain_variance + obs_variance)) * beta

        .. :math::

            Mean(t) ≡ E(beta_{t,w} | beta_ˆ 1:T )
            = fwd_mean[t - 1] + (obs_variance / fwd_variance[t - 1] + obs_variance) +
            (1 - obs_variance / fwd_variance[t - 1] + obs_variance)) * mean[t]

        Parameters
        ----------
        word: int
            The word's ID.
        chain_variance : float
            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            The first returned value is the mean of each word in each time slice, the second value is the
            inferred posterior mean for the same pairs.

        """
        T = self.num_time_slices
        obs = self.obs[word]
        fwd_variance = self.fwd_variance[word]
        mean = self.mean[word]
        fwd_mean = self.fwd_mean[word]
        fwd_mean[0] = 0
        for t in range(1, T + 1):
            c = self.obs_variance / (fwd_variance[t - 1] + chain_variance + self.obs_variance)
            fwd_mean[t] = c * fwd_mean[t - 1] + (1 - c) * obs[t - 1]
        mean[T] = fwd_mean[T]
        for t in range(T - 1, -1, -1):
            if chain_variance == 0.0:
                c = 0.0
            else:
                c = chain_variance / (fwd_variance[t] + chain_variance)
            mean[t] = c * fwd_mean[t] + (1 - c) * mean[t + 1]
        return (mean, fwd_mean)

    def compute_expected_log_prob(self):
        """Compute the expected log probability given values of m.

        The appendix describes the Expectation of log-probabilities in equation 5 of the DTM paper;
        The below implementation is the result of solving the equation and is implemented as in the original
        Blei DTM code.

        Returns
        -------
        numpy.ndarray of float
            The expected value for the log probabilities for each word and time slice.

        """
        for (w, t), val in np.ndenumerate(self.e_log_prob):
            self.e_log_prob[w][t] = self.mean[w][t + 1] - np.log(self.zeta[t])
        return self.e_log_prob

    def sslm_counts_init(self, obs_variance, chain_variance, sstats):
        """Initialize the State Space Language Model with LDA sufficient statistics.

        Called for each topic-chain and initializes initial mean, variance and Topic-Word probabilities
        for the first time-slice.

        Parameters
        ----------
        obs_variance : float, optional
            Observed variance used to approximate the true and forward variance.
        chain_variance : float
            Gaussian parameter defined in the beta distribution to dictate how the beta values evolve over time.
        sstats : numpy.ndarray
            Sufficient statistics of the LDA model. Corresponds to matrix beta in the linked paper for time slice 0,
            expected shape (`self.vocab_len`, `num_topics`).

        """
        W = self.vocab_len
        T = self.num_time_slices
        log_norm_counts = np.copy(sstats)
        log_norm_counts /= sum(log_norm_counts)
        log_norm_counts += 1.0 / W
        log_norm_counts /= sum(log_norm_counts)
        log_norm_counts = np.log(log_norm_counts)
        self.obs = np.repeat(log_norm_counts, T, axis=0).reshape(W, T)
        self.obs_variance = obs_variance
        self.chain_variance = chain_variance
        for w in range(W):
            self.variance[w], self.fwd_variance[w] = self.compute_post_variance(w, self.chain_variance)
            self.mean[w], self.fwd_mean[w] = self.compute_post_mean(w, self.chain_variance)
        self.zeta = self.update_zeta()
        self.e_log_prob = self.compute_expected_log_prob()

    def fit_sslm(self, sstats):
        """Fits variational distribution.

        This is essentially the m-step.
        Maximizes the approximation of the true posterior for a particular topic using the provided sufficient
        statistics. Updates the values using :meth:`~gensim.models.ldaseqmodel.sslm.update_obs` and
        :meth:`~gensim.models.ldaseqmodel.sslm.compute_expected_log_prob`.

        Parameters
        ----------
        sstats : numpy.ndarray
            Sufficient statistics for a particular topic. Corresponds to matrix beta in the linked paper for the
            current time slice, expected shape (`self.vocab_len`, `num_topics`).

        Returns
        -------
        float
            The lower bound for the true posterior achieved using the fitted approximate distribution.

        """
        W = self.vocab_len
        bound = 0
        old_bound = 0
        sslm_fit_threshold = 1e-06
        sslm_max_iter = 2
        converged = sslm_fit_threshold + 1
        self.variance, self.fwd_variance = (np.array(x) for x in zip(*(self.compute_post_variance(w, self.chain_variance) for w in range(W))))
        totals = sstats.sum(axis=0)
        iter_ = 0
        model = 'DTM'
        if model == 'DTM':
            bound = self.compute_bound(sstats, totals)
        if model == 'DIM':
            bound = self.compute_bound_fixed(sstats, totals)
        logger.info('initial sslm bound is %f', bound)
        while converged > sslm_fit_threshold and iter_ < sslm_max_iter:
            iter_ += 1
            old_bound = bound
            self.obs, self.zeta = self.update_obs(sstats, totals)
            if model == 'DTM':
                bound = self.compute_bound(sstats, totals)
            if model == 'DIM':
                bound = self.compute_bound_fixed(sstats, totals)
            converged = np.fabs((bound - old_bound) / old_bound)
            logger.info('iteration %i iteration lda seq bound is %f convergence is %f', iter_, bound, converged)
        self.e_log_prob = self.compute_expected_log_prob()
        return bound

    def compute_bound(self, sstats, totals):
        """Compute the maximized lower bound achieved for the log probability of the true posterior.

        Uses the formula presented in the appendix of the DTM paper (formula no. 5).

        Parameters
        ----------
        sstats : numpy.ndarray
            Sufficient statistics for a particular topic. Corresponds to matrix beta in the linked paper for the first
            time slice, expected shape (`self.vocab_len`, `num_topics`).
        totals : list of int of length `len(self.time_slice)`
            The totals for each time slice.

        Returns
        -------
        float
            The maximized lower bound.

        """
        w = self.vocab_len
        t = self.num_time_slices
        term_1 = 0
        term_2 = 0
        term_3 = 0
        val = 0
        ent = 0
        chain_variance = self.chain_variance
        self.mean, self.fwd_mean = (np.array(x) for x in zip(*(self.compute_post_mean(w, self.chain_variance) for w in range(w))))
        self.zeta = self.update_zeta()
        val = sum((self.variance[w][0] - self.variance[w][t] for w in range(w))) / 2 * chain_variance
        logger.info('Computing bound, all times')
        for t in range(1, t + 1):
            term_1 = 0.0
            term_2 = 0.0
            ent = 0.0
            for w in range(w):
                m = self.mean[w][t]
                prev_m = self.mean[w][t - 1]
                v = self.variance[w][t]
                term_1 += np.power(m - prev_m, 2) / (2 * chain_variance) - v / chain_variance - np.log(chain_variance)
                term_2 += sstats[w][t - 1] * m
                ent += np.log(v) / 2
            term_3 = -totals[t - 1] * np.log(self.zeta[t - 1])
            val += term_2 + term_3 + ent - term_1
        return val

    def update_obs(self, sstats, totals):
        """Optimize the bound with respect to the observed variables.

        TODO:
        This is by far the slowest function in the whole algorithm.
        Replacing or improving the performance of this would greatly speed things up.

        Parameters
        ----------
        sstats : numpy.ndarray
            Sufficient statistics for a particular topic. Corresponds to matrix beta in the linked paper for the first
            time slice, expected shape (`self.vocab_len`, `num_topics`).
        totals : list of int of length `len(self.time_slice)`
            The totals for each time slice.

        Returns
        -------
        (numpy.ndarray of float, numpy.ndarray of float)
            The updated optimized values for obs and the zeta variational parameter.

        """
        OBS_NORM_CUTOFF = 2
        STEP_SIZE = 0.01
        TOL = 0.001
        W = self.vocab_len
        T = self.num_time_slices
        runs = 0
        mean_deriv_mtx = np.zeros((T, T + 1))
        norm_cutoff_obs = None
        for w in range(W):
            w_counts = sstats[w]
            counts_norm = 0
            for i in range(len(w_counts)):
                counts_norm += w_counts[i] * w_counts[i]
            counts_norm = np.sqrt(counts_norm)
            if counts_norm < OBS_NORM_CUTOFF and norm_cutoff_obs is not None:
                obs = self.obs[w]
                norm_cutoff_obs = np.copy(obs)
            else:
                if counts_norm < OBS_NORM_CUTOFF:
                    w_counts = np.zeros(len(w_counts))
                for t in range(T):
                    mean_deriv_mtx[t] = self.compute_mean_deriv(w, t, mean_deriv_mtx[t])
                deriv = np.zeros(T)
                args = (self, w_counts, totals, mean_deriv_mtx, w, deriv)
                obs = self.obs[w]
                model = 'DTM'
                if model == 'DTM':
                    obs = optimize.fmin_cg(f=f_obs, fprime=df_obs, x0=obs, gtol=TOL, args=args, epsilon=STEP_SIZE, disp=0)
                if model == 'DIM':
                    pass
                runs += 1
                if counts_norm < OBS_NORM_CUTOFF:
                    norm_cutoff_obs = obs
                self.obs[w] = obs
        self.zeta = self.update_zeta()
        return (self.obs, self.zeta)

    def compute_mean_deriv(self, word, time, deriv):
        """Helper functions for optimizing a function.

        Compute the derivative of:

        .. :math::

            E[\x08eta_{t,w}]/d obs_{s,w} for t = 1:T.

        Parameters
        ----------
        word : int
            The word's ID.
        time : int
            The time slice.
        deriv : list of float
            Derivative for each time slice.

        Returns
        -------
        list of float
            Mean derivative for each time slice.

        """
        T = self.num_time_slices
        fwd_variance = self.variance[word]
        deriv[0] = 0
        for t in range(1, T + 1):
            if self.obs_variance > 0.0:
                w = self.obs_variance / (fwd_variance[t - 1] + self.chain_variance + self.obs_variance)
            else:
                w = 0.0
            val = w * deriv[t - 1]
            if time == t - 1:
                val += 1 - w
            deriv[t] = val
        for t in range(T - 1, -1, -1):
            if self.chain_variance == 0.0:
                w = 0.0
            else:
                w = self.chain_variance / (fwd_variance[t] + self.chain_variance)
            deriv[t] = w * deriv[t] + (1 - w) * deriv[t + 1]
        return deriv

    def compute_obs_deriv(self, word, word_counts, totals, mean_deriv_mtx, deriv):
        """Derivation of obs which is used in derivative function `df_obs` while optimizing.

        Parameters
        ----------
        word : int
            The word's ID.
        word_counts : list of int
            Total word counts for each time slice.
        totals : list of int of length `len(self.time_slice)`
            The totals for each time slice.
        mean_deriv_mtx : list of float
            Mean derivative for each time slice.
        deriv : list of float
            Mean derivative for each time slice.

        Returns
        -------
        list of float
            Mean derivative for each time slice.

        """
        init_mult = 1000
        T = self.num_time_slices
        mean = self.mean[word]
        variance = self.variance[word]
        self.temp_vect = np.zeros(T)
        for u in range(T):
            self.temp_vect[u] = np.exp(mean[u + 1] + variance[u + 1] / 2)
        for t in range(T):
            mean_deriv = mean_deriv_mtx[t]
            term1 = 0
            term2 = 0
            term3 = 0
            term4 = 0
            for u in range(1, T + 1):
                mean_u = mean[u]
                mean_u_prev = mean[u - 1]
                dmean_u = mean_deriv[u]
                dmean_u_prev = mean_deriv[u - 1]
                term1 += (mean_u - mean_u_prev) * (dmean_u - dmean_u_prev)
                term2 += (word_counts[u - 1] - totals[u - 1] * self.temp_vect[u - 1] / self.zeta[u - 1]) * dmean_u
                model = 'DTM'
                if model == 'DIM':
                    pass
            if self.chain_variance:
                term1 = -(term1 / self.chain_variance)
                term1 = term1 - mean[0] * mean_deriv[0] / (init_mult * self.chain_variance)
            else:
                term1 = 0.0
            deriv[t] = term1 + term2 + term3 + term4
        return deriv