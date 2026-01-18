import codecs
import logging
import os
import pickle
import re
import shutil
import tarfile
from contextlib import suppress
import joblib
import numpy as np
import scipy.sparse as sp
from .. import preprocessing
from ..feature_extraction.text import CountVectorizer
from ..utils import Bunch, check_random_state
from ..utils._param_validation import StrOptions, validate_params
from . import get_data_home, load_files
from ._base import (
Load and vectorize the 20 newsgroups dataset (classification).

    Download it if necessary.

    This is a convenience function; the transformation is done using the
    default settings for
    :class:`~sklearn.feature_extraction.text.CountVectorizer`. For more
    advanced usage (stopword filtering, n-gram extraction, etc.), combine
    fetch_20newsgroups with a custom
    :class:`~sklearn.feature_extraction.text.CountVectorizer`,
    :class:`~sklearn.feature_extraction.text.HashingVectorizer`,
    :class:`~sklearn.feature_extraction.text.TfidfTransformer` or
    :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.

    The resulting counts are normalized using
    :func:`sklearn.preprocessing.normalize` unless normalize is set to False.

    =================   ==========
    Classes                     20
    Samples total            18846
    Dimensionality          130107
    Features                  real
    =================   ==========

    Read more in the :ref:`User Guide <20newsgroups_dataset>`.

    Parameters
    ----------
    subset : {'train', 'test', 'all'}, default='train'
        Select the dataset to load: 'train' for the training set, 'test'
        for the test set, 'all' for both, with shuffled ordering.

    remove : tuple, default=()
        May contain any subset of ('headers', 'footers', 'quotes'). Each of
        these are kinds of text that will be detected and removed from the
        newsgroup posts, preventing classifiers from overfitting on
        metadata.

        'headers' removes newsgroup headers, 'footers' removes blocks at the
        ends of posts that look like signatures, and 'quotes' removes lines
        that appear to be quoting another post.

    data_home : str or path-like, default=None
        Specify an download and cache folder for the datasets. If None,
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.

    return_X_y : bool, default=False
        If True, returns ``(data.data, data.target)`` instead of a Bunch
        object.

        .. versionadded:: 0.20

    normalize : bool, default=True
        If True, normalizes each document's feature vector to unit norm using
        :func:`sklearn.preprocessing.normalize`.

        .. versionadded:: 0.22

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric, string, or categorical). The target is
        a pandas DataFrame or Series depending on the number of
        `target_columns`.

        .. versionadded:: 0.24

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data: {sparse matrix, dataframe} of shape (n_samples, n_features)
            The input data matrix. If ``as_frame`` is `True`, ``data`` is
            a pandas DataFrame with sparse columns.
        target: {ndarray, series} of shape (n_samples,)
            The target labels. If ``as_frame`` is `True`, ``target`` is a
            pandas Series.
        target_names: list of shape (n_classes,)
            The names of target classes.
        DESCR: str
            The full description of the dataset.
        frame: dataframe of shape (n_samples, n_features + 1)
            Only present when `as_frame=True`. Pandas DataFrame with ``data``
            and ``target``.

            .. versionadded:: 0.24

    (data, target) : tuple if ``return_X_y`` is True
        `data` and `target` would be of the format defined in the `Bunch`
        description above.

        .. versionadded:: 0.20
    