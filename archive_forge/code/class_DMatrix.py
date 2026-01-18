import logging
from typing import Dict, Optional
import xgboost as xgb
import modin.pandas as pd
from modin.config import Engine
from modin.distributed.dataframe.pandas import unwrap_partitions
class DMatrix:
    """
    DMatrix holds references to partitions of Modin DataFrame.

    On init stage unwrapping partitions of Modin DataFrame is started.

    Parameters
    ----------
    data : modin.pandas.DataFrame
        Data source of DMatrix.
    label : modin.pandas.DataFrame or modin.pandas.Series, optional
        Labels used for training.
    missing : float, optional
        Value in the input data which needs to be present as a missing
        value. If ``None``, defaults to ``np.nan``.
    silent : boolean, optional
        Whether to print messages during construction or not.
    feature_names : list, optional
        Set names for features.
    feature_types : list, optional
        Set types for features.
    feature_weights : array_like, optional
        Set feature weights for column sampling.
    enable_categorical : boolean, optional
        Experimental support of specializing for categorical features.

    Notes
    -----
    Currently DMatrix doesn't support `weight`, `base_margin`, `nthread`,
    `group`, `qid`, `label_lower_bound`, `label_upper_bound` parameters.
    """

    def __init__(self, data, label=None, missing=None, silent=False, feature_names=None, feature_types=None, feature_weights=None, enable_categorical=None):
        assert isinstance(data, pd.DataFrame), f'Type of `data` is {type(data)}, but expected {pd.DataFrame}.'
        if label is not None:
            assert isinstance(label, (pd.DataFrame, pd.Series)), f'Type of `data` is {type(label)}, but expected {pd.DataFrame} or {pd.Series}.'
            self.label = unwrap_partitions(label, axis=0)
        else:
            self.label = None
        self.data = unwrap_partitions(data, axis=0, get_ip=True)
        self._n_rows = data.shape[0]
        self._n_cols = data.shape[1]
        for i, dtype in enumerate(data.dtypes):
            if dtype == 'object':
                raise ValueError(f'Column {i} has unsupported data type {dtype}.')
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.missing = missing
        self.silent = silent
        self.feature_weights = feature_weights
        self.enable_categorical = enable_categorical
        self.metadata = (data.index, data.columns, data._query_compiler._modin_frame.row_lengths)

    def __iter__(self):
        """
        Return unwrapped `self.data` and `self.label`.

        Yields
        ------
        list
            List of `self.data` with pairs of references to IP of row partition
            and row partition [(IP_ref0, partition_ref0), ..].
        list
            List of `self.label` with references to row partitions
            [partition_ref0, ..].
        """
        yield self.data
        yield self.label

    def get_dmatrix_params(self):
        """
        Get dict of DMatrix parameters excluding `self.data`/`self.label`.

        Returns
        -------
        dict
        """
        dmatrix_params = {'feature_names': self.feature_names, 'feature_types': self.feature_types, 'missing': self.missing, 'silent': self.silent, 'feature_weights': self.feature_weights, 'enable_categorical': self.enable_categorical}
        return dmatrix_params

    @property
    def feature_names(self):
        """
        Get column labels.

        Returns
        -------
        Column labels.
        """
        return self._feature_names

    @feature_names.setter
    def feature_names(self, feature_names):
        """
        Set column labels.

        Parameters
        ----------
        feature_names : list or None
            Labels for columns. In the case of ``None``, existing feature names will be reset.
        """
        if feature_names is not None:
            feature_names = list(feature_names) if not isinstance(feature_names, str) else [feature_names]
            if len(feature_names) != len(set(feature_names)):
                raise ValueError('Items in `feature_names` must be unique.')
            if len(feature_names) != self.num_col() and self.num_col() != 0:
                raise ValueError('`feature_names` must have the same width as `self.data`.')
            if not all((isinstance(f, str) and (not any((x in f for x in set(('[', ']', '<'))))) for f in feature_names)):
                raise ValueError('Items of `feature_names` must be string and must not contain [, ] or <.')
        else:
            feature_names = None
        self._feature_names = feature_names

    @property
    def feature_types(self):
        """
        Get column types.

        Returns
        -------
        Column types.
        """
        return self._feature_types

    @feature_types.setter
    def feature_types(self, feature_types):
        """
        Set column types.

        Parameters
        ----------
        feature_types : list or None
            Labels for columns. In case None, existing feature names will be reset.
        """
        if feature_types is not None:
            if not isinstance(feature_types, (list, str)):
                raise TypeError('feature_types must be string or list of strings')
            if isinstance(feature_types, str):
                feature_types = [feature_types] * self.num_col()
                feature_types = list(feature_types) if not isinstance(feature_types, str) else [feature_types]
        else:
            feature_types = None
        self._feature_types = feature_types

    def num_row(self):
        """
        Get number of rows.

        Returns
        -------
        int
        """
        return self._n_rows

    def num_col(self):
        """
        Get number of columns.

        Returns
        -------
        int
        """
        return self._n_cols

    def get_float_info(self, name):
        """
        Get float property from the DMatrix.

        Parameters
        ----------
        name : str
            The field name of the information.

        Returns
        -------
        A NumPy array of float information of the data.
        """
        return getattr(self, name)

    def set_info(self, *, label=None, feature_names=None, feature_types=None, feature_weights=None) -> None:
        """
        Set meta info for DMatrix.

        Parameters
        ----------
        label : modin.pandas.DataFrame or modin.pandas.Series, optional
            Labels used for training.
        feature_names : list, optional
            Set names for features.
        feature_types : list, optional
            Set types for features.
        feature_weights : array_like, optional
            Set feature weights for column sampling.
        """
        if label is not None:
            self.label = label
        if feature_names is not None:
            self.feature_names = feature_names
        if feature_types is not None:
            self.feature_types = feature_types
        if feature_weights is not None:
            self.feature_weights = feature_weights