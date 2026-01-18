import importlib
from functools import wraps
from typing import Protocol, runtime_checkable
import numpy as np
from scipy.sparse import issparse
from .._config import get_config
from ._available_if import available_if
@runtime_checkable
class ContainerAdapterProtocol(Protocol):
    container_lib: str

    def create_container(self, X_output, X_original, columns, inplace=False):
        """Create container from `X_output` with additional metadata.

        Parameters
        ----------
        X_output : {ndarray, dataframe}
            Data to wrap.

        X_original : {ndarray, dataframe}
            Original input dataframe. This is used to extract the metadata that should
            be passed to `X_output`, e.g. pandas row index.

        columns : callable, ndarray, or None
            The column names or a callable that returns the column names. The
            callable is useful if the column names require some computation. If `None`,
            then no columns are passed to the container's constructor.

        inplace : bool, default=False
            Whether or not we intend to modify `X_output` in-place. However, it does
            not guarantee that we return the same object if the in-place operation
            is not possible.

        Returns
        -------
        wrapped_output : container_type
            `X_output` wrapped into the container type.
        """

    def is_supported_container(self, X):
        """Return True if X is a supported container.

        Parameters
        ----------
        Xs: container
            Containers to be checked.

        Returns
        -------
        is_supported_container : bool
            True if X is a supported container.
        """

    def rename_columns(self, X, columns):
        """Rename columns in `X`.

        Parameters
        ----------
        X : container
            Container which columns is updated.

        columns : ndarray of str
            Columns to update the `X`'s columns with.

        Returns
        -------
        updated_container : container
            Container with new names.
        """

    def hstack(self, Xs):
        """Stack containers horizontally (column-wise).

        Parameters
        ----------
        Xs : list of containers
            List of containers to stack.

        Returns
        -------
        stacked_Xs : container
            Stacked containers.
        """