from statsmodels.base.elastic_net import RegularizedResults
from statsmodels.stats.regularized_covariance import _calc_nodewise_row, \
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
import numpy as np
class DistributedModel:
    __doc__ = '\n    Distributed model class\n\n    Parameters\n    ----------\n    partitions : scalar\n        The number of partitions that the data will be split into.\n    model_class : statsmodels model class\n        The model class which will be used for estimation. If None\n        this defaults to OLS.\n    init_kwds : dict-like or None\n        Keywords needed for initializing the model, in addition to\n        endog and exog.\n    init_kwds_generator : generator or None\n        Additional keyword generator that produces model init_kwds\n        that may vary based on data partition.  The current usecase\n        is for WLS and GLS\n    estimation_method : function or None\n        The method that performs the estimation for each partition.\n        If None this defaults to _est_regularized_debiased.\n    estimation_kwds : dict-like or None\n        Keywords to be passed to estimation_method.\n    join_method : function or None\n        The method used to recombine the results from each partition.\n        If None this defaults to _join_debiased.\n    join_kwds : dict-like or None\n        Keywords to be passed to join_method.\n    results_class : results class or None\n        The class of results that should be returned.  If None this\n        defaults to RegularizedResults.\n    results_kwds : dict-like or None\n        Keywords to be passed to results class.\n\n    Attributes\n    ----------\n    partitions : scalar\n        See Parameters.\n    model_class : statsmodels model class\n        See Parameters.\n    init_kwds : dict-like\n        See Parameters.\n    init_kwds_generator : generator or None\n        See Parameters.\n    estimation_method : function\n        See Parameters.\n    estimation_kwds : dict-like\n        See Parameters.\n    join_method : function\n        See Parameters.\n    join_kwds : dict-like\n        See Parameters.\n    results_class : results class\n        See Parameters.\n    results_kwds : dict-like\n        See Parameters.\n\n    Notes\n    -----\n\n    Examples\n    --------\n    '

    def __init__(self, partitions, model_class=None, init_kwds=None, estimation_method=None, estimation_kwds=None, join_method=None, join_kwds=None, results_class=None, results_kwds=None):
        self.partitions = partitions
        if model_class is None:
            self.model_class = OLS
        else:
            self.model_class = model_class
        if init_kwds is None:
            self.init_kwds = {}
        else:
            self.init_kwds = init_kwds
        if estimation_method is None:
            self.estimation_method = _est_regularized_debiased
        else:
            self.estimation_method = estimation_method
        if estimation_kwds is None:
            self.estimation_kwds = {}
        else:
            self.estimation_kwds = estimation_kwds
        if join_method is None:
            self.join_method = _join_debiased
        else:
            self.join_method = join_method
        if join_kwds is None:
            self.join_kwds = {}
        else:
            self.join_kwds = join_kwds
        if results_class is None:
            self.results_class = RegularizedResults
        else:
            self.results_class = results_class
        if results_kwds is None:
            self.results_kwds = {}
        else:
            self.results_kwds = results_kwds

    def fit(self, data_generator, fit_kwds=None, parallel_method='sequential', parallel_backend=None, init_kwds_generator=None):
        """Performs the distributed estimation using the corresponding
        DistributedModel

        Parameters
        ----------
        data_generator : generator
            A generator that produces a sequence of tuples where the first
            element in the tuple corresponds to an endog array and the
            element corresponds to an exog array.
        fit_kwds : dict-like or None
            Keywords needed for the model fitting.
        parallel_method : str
            type of distributed estimation to be used, currently
            "sequential", "joblib" and "dask" are supported.
        parallel_backend : None or joblib parallel_backend object
            used to allow support for more complicated backends,
            ex: dask.distributed
        init_kwds_generator : generator or None
            Additional keyword generator that produces model init_kwds
            that may vary based on data partition.  The current usecase
            is for WLS and GLS

        Returns
        -------
        join_method result.  For the default, _join_debiased, it returns a
        p length array.
        """
        if fit_kwds is None:
            fit_kwds = {}
        if parallel_method == 'sequential':
            results_l = self.fit_sequential(data_generator, fit_kwds, init_kwds_generator)
        elif parallel_method == 'joblib':
            results_l = self.fit_joblib(data_generator, fit_kwds, parallel_backend, init_kwds_generator)
        else:
            raise ValueError('parallel_method: %s is currently not supported' % parallel_method)
        params = self.join_method(results_l, **self.join_kwds)
        res_mod = self.model_class([0], [0], **self.init_kwds)
        return self.results_class(res_mod, params, **self.results_kwds)

    def fit_sequential(self, data_generator, fit_kwds, init_kwds_generator=None):
        """Sequentially performs the distributed estimation using
        the corresponding DistributedModel

        Parameters
        ----------
        data_generator : generator
            A generator that produces a sequence of tuples where the first
            element in the tuple corresponds to an endog array and the
            element corresponds to an exog array.
        fit_kwds : dict-like
            Keywords needed for the model fitting.
        init_kwds_generator : generator or None
            Additional keyword generator that produces model init_kwds
            that may vary based on data partition.  The current usecase
            is for WLS and GLS

        Returns
        -------
        join_method result.  For the default, _join_debiased, it returns a
        p length array.
        """
        results_l = []
        if init_kwds_generator is None:
            for pnum, (endog, exog) in enumerate(data_generator):
                results = _helper_fit_partition(self, pnum, endog, exog, fit_kwds)
                results_l.append(results)
        else:
            tup_gen = enumerate(zip(data_generator, init_kwds_generator))
            for pnum, ((endog, exog), init_kwds_e) in tup_gen:
                results = _helper_fit_partition(self, pnum, endog, exog, fit_kwds, init_kwds_e)
                results_l.append(results)
        return results_l

    def fit_joblib(self, data_generator, fit_kwds, parallel_backend, init_kwds_generator=None):
        """Performs the distributed estimation in parallel using joblib

        Parameters
        ----------
        data_generator : generator
            A generator that produces a sequence of tuples where the first
            element in the tuple corresponds to an endog array and the
            element corresponds to an exog array.
        fit_kwds : dict-like
            Keywords needed for the model fitting.
        parallel_backend : None or joblib parallel_backend object
            used to allow support for more complicated backends,
            ex: dask.distributed
        init_kwds_generator : generator or None
            Additional keyword generator that produces model init_kwds
            that may vary based on data partition.  The current usecase
            is for WLS and GLS

        Returns
        -------
        join_method result.  For the default, _join_debiased, it returns a
        p length array.
        """
        from statsmodels.tools.parallel import parallel_func
        par, f, n_jobs = parallel_func(_helper_fit_partition, self.partitions)
        if parallel_backend is None and init_kwds_generator is None:
            results_l = par((f(self, pnum, endog, exog, fit_kwds) for pnum, (endog, exog) in enumerate(data_generator)))
        elif parallel_backend is not None and init_kwds_generator is None:
            with parallel_backend:
                results_l = par((f(self, pnum, endog, exog, fit_kwds) for pnum, (endog, exog) in enumerate(data_generator)))
        elif parallel_backend is None and init_kwds_generator is not None:
            tup_gen = enumerate(zip(data_generator, init_kwds_generator))
            results_l = par((f(self, pnum, endog, exog, fit_kwds, init_kwds) for pnum, ((endog, exog), init_kwds) in tup_gen))
        elif parallel_backend is not None and init_kwds_generator is not None:
            tup_gen = enumerate(zip(data_generator, init_kwds_generator))
            with parallel_backend:
                results_l = par((f(self, pnum, endog, exog, fit_kwds, init_kwds) for pnum, ((endog, exog), init_kwds) in tup_gen))
        return results_l