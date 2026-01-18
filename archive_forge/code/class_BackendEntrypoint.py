from __future__ import annotations
import logging
import os
import time
import traceback
from collections.abc import Iterable
from glob import glob
from typing import TYPE_CHECKING, Any, ClassVar
import numpy as np
from xarray.conventions import cf_encoder
from xarray.core import indexing
from xarray.core.utils import FrozenDict, NdimSizeLenMixin, is_remote_uri
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
class BackendEntrypoint:
    """
    ``BackendEntrypoint`` is a class container and it is the main interface
    for the backend plugins, see :ref:`RST backend_entrypoint`.
    It shall implement:

    - ``open_dataset`` method: it shall implement reading from file, variables
      decoding and it returns an instance of :py:class:`~xarray.Dataset`.
      It shall take in input at least ``filename_or_obj`` argument and
      ``drop_variables`` keyword argument.
      For more details see :ref:`RST open_dataset`.
    - ``guess_can_open`` method: it shall return ``True`` if the backend is able to open
      ``filename_or_obj``, ``False`` otherwise. The implementation of this
      method is not mandatory.
    - ``open_datatree`` method: it shall implement reading from file, variables
      decoding and it returns an instance of :py:class:`~datatree.DataTree`.
      It shall take in input at least ``filename_or_obj`` argument. The
      implementation of this method is not mandatory.  For more details see
      <reference to open_datatree documentation>.

    Attributes
    ----------

    open_dataset_parameters : tuple, default: None
        A list of ``open_dataset`` method parameters.
        The setting of this attribute is not mandatory.
    description : str, default: ""
        A short string describing the engine.
        The setting of this attribute is not mandatory.
    url : str, default: ""
        A string with the URL to the backend's documentation.
        The setting of this attribute is not mandatory.
    """
    open_dataset_parameters: ClassVar[tuple | None] = None
    description: ClassVar[str] = ''
    url: ClassVar[str] = ''

    def __repr__(self) -> str:
        txt = f'<{type(self).__name__}>'
        if self.description:
            txt += f'\n  {self.description}'
        if self.url:
            txt += f'\n  Learn more at {self.url}'
        return txt

    def open_dataset(self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore, *, drop_variables: str | Iterable[str] | None=None, **kwargs: Any) -> Dataset:
        """
        Backend open_dataset method used by Xarray in :py:func:`~xarray.open_dataset`.
        """
        raise NotImplementedError()

    def guess_can_open(self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore) -> bool:
        """
        Backend open_dataset method used by Xarray in :py:func:`~xarray.open_dataset`.
        """
        return False

    def open_datatree(self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore, **kwargs: Any) -> DataTree:
        """
        Backend open_datatree method used by Xarray in :py:func:`~xarray.open_datatree`.
        """
        raise NotImplementedError()