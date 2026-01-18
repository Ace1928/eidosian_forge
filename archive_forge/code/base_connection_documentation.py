from __future__ import annotations
import json
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
from streamlit.runtime.secrets import AttrDict, secrets_singleton
from streamlit.util import calc_md5
Create an instance of an underlying connection object.

        This abstract method is the one method that we require subclasses of
        BaseConnection to provide an implementation for. It is called when first
        creating a connection and when reconnecting after a connection is reset.

        Parameters
        ----------
        kwargs : dict

        Returns
        -------
        RawConnectionT
            The underlying connection object.
        