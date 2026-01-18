from statsmodels.compat.pandas import MONTH_END, QUARTER_END
from collections import OrderedDict
from warnings import warn
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.validation import int_like
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.multivariate.pca import PCA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace._quarterly_ar1 import QuarterlyAR1
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import string_like
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.statespace import mlemodel, initialization
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.base.data import PandasData
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.tableformatting import fmt_params
class DynamicFactorMQStates(dict):
    """
    Helper class for describing and indexing the state vector.

    Parameters
    ----------
    k_endog_M : int
        Number of monthly (or non-time-specific, if k_endog_Q=0) variables.
    k_endog_Q : int
        Number of quarterly variables.
    endog_names : list
        Names of the endogenous variables.
    factors : int, list, or dict
        Integer giving the number of (global) factors, a list with the names of
        (global) factors, or a dictionary with:

        - keys : names of endogenous variables
        - values : lists of factor names.

        If this is an integer, then the factor names will be 0, 1, ....
    factor_orders : int or dict
        Integer describing the order of the vector autoregression (VAR)
        governing all factor block dynamics or dictionary with:

        - keys : factor name or tuples of factor names in a block
        - values : integer describing the VAR order for that factor block

        If a dictionary, this defines the order of the factor blocks in the
        state vector. Otherwise, factors are ordered so that factors that load
        on more variables come first (and then alphabetically, to break ties).
    factor_multiplicities : int or dict
        This argument provides a convenient way to specify multiple factors
        that load identically on variables. For example, one may want two
        "global" factors (factors that load on all variables) that evolve
        jointly according to a VAR. One could specify two global factors in the
        `factors` argument and specify that they are in the same block in the
        `factor_orders` argument, but it is easier to specify a single global
        factor in the `factors` argument, and set the order in the
        `factor_orders` argument, and then set the factor multiplicity to 2.

        This argument must be an integer describing the factor multiplicity for
        all factors or dictionary with:

        - keys : factor name
        - values : integer describing the factor multiplicity for the factors
          in the given block
    idiosyncratic_ar1 : bool
        Whether or not to model the idiosyncratic component for each series as
        an AR(1) process. If False, the idiosyncratic component is instead
        modeled as white noise.

    Attributes
    ----------
    k_endog : int
        Total number of endogenous variables.
    k_states : int
        Total number of state variables (those associated with the factors and
        those associated with the idiosyncratic disturbances).
    k_posdef : int
        Total number of state disturbance terms (those associated with the
        factors and those associated with the idiosyncratic disturbances).
    k_endog_M : int
        Number of monthly (or non-time-specific, if k_endog_Q=0) variables.
    k_endog_Q : int
        Number of quarterly variables.
    k_factors : int
        Total number of factors. Note that factor multiplicities will have
        already been expanded.
    k_states_factors : int
        The number of state variables associated with factors (includes both
        factors and lags of factors included in the state vector).
    k_posdef_factors : int
        The number of state disturbance terms associated with factors.
    k_states_idio : int
        Total number of state variables associated with idiosyncratic
        disturbances.
    k_posdef_idio : int
        Total number of state disturbance terms associated with idiosyncratic
        disturbances.
    k_states_idio_M : int
        The number of state variables associated with idiosyncratic
        disturbances for monthly (or non-time-specific if there are no
        quarterly variables) variables. If the disturbances are AR(1), then
        this will be equal to `k_endog_M`, otherwise it will be equal to zero.
    k_states_idio_Q : int
        The number of state variables associated with idiosyncratic
        disturbances for quarterly variables. This will always be equal to
        `k_endog_Q * 5`, even if the disturbances are not AR(1).
    k_posdef_idio_M : int
        The number of state disturbance terms associated with idiosyncratic
        disturbances for monthly (or non-time-specific if there are no
        quarterly variables) variables. If the disturbances are AR(1), then
        this will be equal to `k_endog_M`, otherwise it will be equal to zero.
    k_posdef_idio_Q : int
        The number of state disturbance terms associated with idiosyncratic
        disturbances for quarterly variables. This will always be equal to
        `k_endog_Q`, even if the disturbances are not AR(1).
    idiosyncratic_ar1 : bool
        Whether or not to model the idiosyncratic component for each series as
        an AR(1) process.
    factor_blocks : list of FactorBlock
        List of `FactorBlock` helper instances for each factor block.
    factor_names : list of str
        List of factor names.
    factors : dict
        Dictionary with:

        - keys : names of endogenous variables
        - values : lists of factor names.

        Note that factor multiplicities will have already been expanded.
    factor_orders : dict
        Dictionary with:

        - keys : tuple of factor names
        - values : integer describing autoregression order

        Note that factor multiplicities will have already been expanded.
    max_factor_order : int
        Maximum autoregression order across all factor blocks.
    factor_block_orders : pd.Series
        Series containing lag orders, with the factor block (a tuple of factor
        names) as the index.
    factor_multiplicities : dict
        Dictionary with:

        - keys : factor name
        - values : integer describing the factor multiplicity for the factors
          in the given block
    endog_factor_map : dict
        Dictionary with:

        - keys : endog name
        - values : list of factor names
    loading_counts : pd.Series
        Series containing number of endogenous variables loading on each
        factor, with the factor name as the index.
    block_loading_counts : dict
        Dictionary with:

        - keys : tuple of factor names
        - values : average number of endogenous variables loading on the block
          (note that average is over the factors in the block)

    Notes
    -----
    The goal of this class is, in particular, to make it easier to retrieve
    indexes of subsets of the state vector.

    Note that the ordering of the factor blocks in the state vector is
    determined by the `factor_orders` argument if a dictionary. Otherwise,
    factors are ordered so that factors that load on more variables come first
    (and then alphabetically, to break ties).

    - `factors_L1` is an array with the indexes of first lag of the factors
      from each block. Ordered first by block, and then by lag.
    - `factors_L1_5` is an array with the indexes contains the first - fifth
      lags of the factors from each block. Ordered first by block, and then by
      lag.
    - `factors_L1_5_ix` is an array shaped (5, k_factors) with the indexes
      of the first - fifth lags of the factors from each block.
    - `idio_ar_L1` is an array with the indexes of the first lag of the
      idiosyncratic AR states, both monthly (if appliable) and quarterly.
    - `idio_ar_M` is a slice with the indexes of the idiosyncratic disturbance
      states for the monthly (or non-time-specific if there are no quarterly
      variables) variables. It is an empty slice if
      `idiosyncratic_ar1 = False`.
    - `idio_ar_Q` is a slice with the indexes of the idiosyncratic disturbance
      states and all lags, for the quarterly variables. It is an empty slice if
      there are no quarterly variable.
    - `idio_ar_Q_ix` is an array shaped (k_endog_Q, 5) with the indexes of the
      first - fifth lags of the idiosyncratic disturbance states for the
      quarterly variables.
    - `endog_factor_iloc` is a list of lists, with entries for each endogenous
      variable. The entry for variable `i`, `endog_factor_iloc[i]` is a list of
      indexes of the factors that variable `i` loads on. This does not include
      any lags, but it can be used with e.g. `factors_L1_5_ix` to get lags.

    """

    def __init__(self, k_endog_M, k_endog_Q, endog_names, factors, factor_orders, factor_multiplicities, idiosyncratic_ar1):
        self.k_endog_M = k_endog_M
        self.k_endog_Q = k_endog_Q
        self.k_endog = self.k_endog_M + self.k_endog_Q
        self.idiosyncratic_ar1 = idiosyncratic_ar1
        factors_is_int = np.issubdtype(type(factors), np.integer)
        factors_is_list = isinstance(factors, (list, tuple))
        orders_is_int = np.issubdtype(type(factor_orders), np.integer)
        if factor_multiplicities is None:
            factor_multiplicities = 1
        mult_is_int = np.issubdtype(type(factor_multiplicities), np.integer)
        if not (factors_is_int or factors_is_list or isinstance(factors, dict)):
            raise ValueError('`factors` argument must an integer number of factors, a list of global factor names, or a dictionary, mapping observed variables to factors.')
        if not (orders_is_int or isinstance(factor_orders, dict)):
            raise ValueError('`factor_orders` argument must either be an integer or a dictionary.')
        if not (mult_is_int or isinstance(factor_multiplicities, dict)):
            raise ValueError('`factor_multiplicities` argument must either be an integer or a dictionary.')
        if factors_is_int or factors_is_list:
            if factors_is_int and factors == 0 or (factors_is_list and len(factors) == 0):
                raise ValueError('The model must contain at least one factor.')
            if factors_is_list:
                factor_names = list(factors)
            else:
                factor_names = [f'{i}' for i in range(factors)]
            factors = {name: factor_names[:] for name in endog_names}
        _factor_names = []
        for val in factors.values():
            _factor_names.extend(val)
        factor_names = set(_factor_names)
        if orders_is_int:
            factor_orders = {factor_name: factor_orders for factor_name in factor_names}
        if mult_is_int:
            factor_multiplicities = {factor_name: factor_multiplicities for factor_name in factor_names}
        factors, factor_orders = self._apply_factor_multiplicities(factors, factor_orders, factor_multiplicities)
        self.factors = factors
        self.factor_orders = factor_orders
        self.factor_multiplicities = factor_multiplicities
        self.endog_factor_map = self._construct_endog_factor_map(factors, endog_names)
        self.k_factors = self.endog_factor_map.shape[1]
        if self.k_factors > self.k_endog_M:
            raise ValueError(f'Number of factors ({self.k_factors}) cannot be greater than the number of monthly endogenous variables ({self.k_endog_M}).')
        self.loading_counts = self.endog_factor_map.sum(axis=0).rename('count').reset_index().sort_values(['count', 'factor'], ascending=[False, True]).set_index('factor')
        block_loading_counts = {block: np.atleast_1d(self.loading_counts.loc[list(block), 'count']).mean(axis=0) for block in factor_orders.keys()}
        ix = pd.Index(block_loading_counts.keys(), tupleize_cols=False, name='block')
        self.block_loading_counts = pd.Series(list(block_loading_counts.values()), index=ix, name='count').to_frame().sort_values(['count', 'block'], ascending=[False, True])['count']
        ix = pd.Index(factor_orders.keys(), tupleize_cols=False, name='block')
        self.factor_block_orders = pd.Series(list(factor_orders.values()), index=ix, name='order')
        if orders_is_int:
            keys = self.block_loading_counts.keys()
            self.factor_block_orders = self.factor_block_orders.loc[keys]
            self.factor_block_orders.index.name = 'block'
        factor_names = pd.Series(np.concatenate(list(self.factor_block_orders.index)))
        missing = [name for name in self.endog_factor_map.columns if name not in factor_names.tolist()]
        if len(missing):
            ix = pd.Index([(factor_name,) for factor_name in missing], tupleize_cols=False, name='block')
            default_block_orders = pd.Series(np.ones(len(ix), dtype=int), index=ix, name='order')
            self.factor_block_orders = self.factor_block_orders.append(default_block_orders)
            factor_names = pd.Series(np.concatenate(list(self.factor_block_orders.index)))
        duplicates = factor_names.duplicated()
        if duplicates.any():
            duplicate_names = set(factor_names[duplicates])
            raise ValueError(f'Each factor can be assigned to at most one block of factors in `factor_orders`. Duplicate entries for {duplicate_names}')
        self.factor_names = factor_names.tolist()
        self.max_factor_order = np.max(self.factor_block_orders)
        self.endog_factor_map = self.endog_factor_map.loc[endog_names, factor_names]
        self.k_states_factors = 0
        self.k_posdef_factors = 0
        state_offset = 0
        self.factor_blocks = []
        for factor_names, factor_order in self.factor_block_orders.items():
            block = FactorBlock(factor_names, factor_order, self.endog_factor_map, state_offset, self.k_endog_Q)
            self.k_states_factors += block.k_states
            self.k_posdef_factors += block.k_factors
            state_offset += block.k_states
            self.factor_blocks.append(block)
        self.k_states_idio_M = self.k_endog_M if idiosyncratic_ar1 else 0
        self.k_states_idio_Q = self.k_endog_Q * 5
        self.k_states_idio = self.k_states_idio_M + self.k_states_idio_Q
        self.k_posdef_idio_M = self.k_endog_M if self.idiosyncratic_ar1 else 0
        self.k_posdef_idio_Q = self.k_endog_Q
        self.k_posdef_idio = self.k_posdef_idio_M + self.k_posdef_idio_Q
        self.k_states = self.k_states_factors + self.k_states_idio
        self.k_posdef = self.k_posdef_factors + self.k_posdef_idio
        self._endog_factor_iloc = None

    def _apply_factor_multiplicities(self, factors, factor_orders, factor_multiplicities):
        """
        Expand `factors` and `factor_orders` to account for factor multiplity.

        For example, if there is a `global` factor with multiplicity 2, then
        this method expands that into `global.1` and `global.2` in both the
        `factors` and `factor_orders` dictionaries.

        Parameters
        ----------
        factors : dict
            Dictionary of {endog_name: list of factor names}
        factor_orders : dict
            Dictionary of {tuple of factor names: factor order}
        factor_multiplicities : dict
            Dictionary of {factor name: factor multiplicity}

        Returns
        -------
        new_factors : dict
            Dictionary of {endog_name: list of factor names}, with factor names
            expanded to incorporate multiplicities.
        new_factors : dict
            Dictionary of {tuple of factor names: factor order}, with factor
            names in each tuple expanded to incorporate multiplicities.
        """
        new_factors = {}
        for endog_name, factors_list in factors.items():
            new_factor_list = []
            for factor_name in factors_list:
                n = factor_multiplicities.get(factor_name, 1)
                if n > 1:
                    new_factor_list += [f'{factor_name}.{i + 1}' for i in range(n)]
                else:
                    new_factor_list.append(factor_name)
            new_factors[endog_name] = new_factor_list
        new_factor_orders = {}
        for block, factor_order in factor_orders.items():
            if not isinstance(block, tuple):
                block = (block,)
            new_block = []
            for factor_name in block:
                n = factor_multiplicities.get(factor_name, 1)
                if n > 1:
                    new_block += [f'{factor_name}.{i + 1}' for i in range(n)]
                else:
                    new_block += [factor_name]
            new_factor_orders[tuple(new_block)] = factor_order
        return (new_factors, new_factor_orders)

    def _construct_endog_factor_map(self, factors, endog_names):
        """
        Construct mapping of observed variables to factors.

        Parameters
        ----------
        factors : dict
            Dictionary of {endog_name: list of factor names}
        endog_names : list of str
            List of the names of the observed variables.

        Returns
        -------
        endog_factor_map : pd.DataFrame
            Boolean dataframe with `endog_names` as the index and the factor
            names (computed from the `factors` input) as the columns. Each cell
            is True if the associated factor is allowed to load on the
            associated observed variable.

        """
        missing = []
        for key, value in factors.items():
            if not isinstance(value, (list, tuple)) or len(value) == 0:
                missing.append(key)
        if len(missing):
            raise ValueError(f'Each observed variable must be mapped to at least one factor in the `factors` dictionary. Variables missing factors are: {missing}.')
        missing = set(endog_names).difference(set(factors.keys()))
        if len(missing):
            raise ValueError(f'If a `factors` dictionary is provided, then it must include entries for each observed variable. Missing variables are: {missing}.')
        factor_names = {}
        for key, value in factors.items():
            if isinstance(value, str):
                factor_names[value] = 0
            else:
                factor_names.update({v: 0 for v in value})
        factor_names = list(factor_names.keys())
        k_factors = len(factor_names)
        endog_factor_map = pd.DataFrame(np.zeros((self.k_endog, k_factors), dtype=bool), index=pd.Index(endog_names, name='endog'), columns=pd.Index(factor_names, name='factor'))
        for key, value in factors.items():
            endog_factor_map.loc[key, value] = True
        return endog_factor_map

    @property
    def factors_L1(self):
        """Factors."""
        ix = np.arange(self.k_states_factors)
        iloc = tuple((ix[block.factors_L1] for block in self.factor_blocks))
        return np.concatenate(iloc)

    @property
    def factors_L1_5_ix(self):
        """Factors plus any lags, index shaped (5, k_factors)."""
        ix = np.arange(self.k_states_factors)
        iloc = []
        for block in self.factor_blocks:
            iloc.append(ix[block.factors_L1_5].reshape(5, block.k_factors))
        return np.concatenate(iloc, axis=1)

    @property
    def idio_ar_L1(self):
        """Idiosyncratic AR states, (first block / lag only)."""
        ix1 = self.k_states_factors
        if self.idiosyncratic_ar1:
            ix2 = ix1 + self.k_endog
        else:
            ix2 = ix1 + self.k_endog_Q
        return np.s_[ix1:ix2]

    @property
    def idio_ar_M(self):
        """Idiosyncratic AR states for monthly variables."""
        ix1 = self.k_states_factors
        ix2 = ix1
        if self.idiosyncratic_ar1:
            ix2 += self.k_endog_M
        return np.s_[ix1:ix2]

    @property
    def idio_ar_Q(self):
        """Idiosyncratic AR states and all lags for quarterly variables."""
        ix1 = self.k_states_factors
        if self.idiosyncratic_ar1:
            ix1 += self.k_endog_M
        ix2 = ix1 + self.k_endog_Q * 5
        return np.s_[ix1:ix2]

    @property
    def idio_ar_Q_ix(self):
        """Idiosyncratic AR (quarterly) state index, (k_endog_Q, lags)."""
        start = self.k_states_factors
        if self.idiosyncratic_ar1:
            start += self.k_endog_M
        return start + np.reshape(np.arange(5 * self.k_endog_Q), (5, self.k_endog_Q)).T

    @property
    def endog_factor_iloc(self):
        """List of list of int, factor indexes for each observed variable."""
        if self._endog_factor_iloc is None:
            ilocs = []
            for i in range(self.k_endog):
                ilocs.append(np.where(self.endog_factor_map.iloc[i])[0])
            self._endog_factor_iloc = ilocs
        return self._endog_factor_iloc

    def __getitem__(self, key):
        """
        Use square brackets to access index / slice elements.

        This is convenient in highlighting the indexing / slice quality of
        these attributes in the code below.
        """
        if key in ['factors_L1', 'factors_L1_5_ix', 'idio_ar_L1', 'idio_ar_M', 'idio_ar_Q', 'idio_ar_Q_ix']:
            return getattr(self, key)
        else:
            raise KeyError(key)