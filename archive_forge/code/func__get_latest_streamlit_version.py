from __future__ import annotations
import random
from importlib.metadata import version as _version
from typing import TYPE_CHECKING, Final
import streamlit.logger as logger
def _get_latest_streamlit_version(timeout: float | None=None) -> Version:
    """Request the latest streamlit version string from PyPI.

    NB: this involves a network call, so it could raise an error
    or take a long time.

    Parameters
    ----------
    timeout : float or None
        The request timeout.

    Returns
    -------
    str
        The version string for the latest version of streamlit
        on PyPI.

    """
    import requests
    rsp = requests.get(PYPI_STREAMLIT_URL, timeout=timeout)
    try:
        version_str = rsp.json()['info']['version']
    except Exception as e:
        raise RuntimeError('Got unexpected response from PyPI', e)
    return _version_str_to_obj(version_str)