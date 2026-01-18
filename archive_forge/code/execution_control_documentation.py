from __future__ import annotations
import os
from typing import Final, NoReturn
import streamlit as st
from streamlit import source_util
from streamlit.deprecation_util import make_deprecated_name_warning
from streamlit.errors import NoSessionContext, StreamlitAPIException
from streamlit.file_util import get_main_script_directory, normalize_path_join
from streamlit.logger import get_logger
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import RerunData, get_script_run_ctx
Programmatically switch the current page in a multipage app.

    When ``st.switch_page`` is called, the current page execution stops and
    the specified page runs as if the user clicked on it in the sidebar
    navigation. The specified page must be recognized by Streamlit's multipage
    architecture (your main Python file or a Python file in a ``pages/``
    folder). Arbitrary Python scripts cannot be passed to ``st.switch_page``.

    Parameters
    ----------
    page: str
        The file path (relative to the main script) of the page to switch to.

    Example
    -------
    Consider the following example given this file structure:

    >>> your-repository/
    >>> ├── pages/
    >>> │   ├── page_1.py
    >>> │   └── page_2.py
    >>> └── your_app.py

    >>> import streamlit as st
    >>>
    >>> if st.button("Home"):
    >>>     st.switch_page("your_app.py")
    >>> if st.button("Page 1"):
    >>>     st.switch_page("pages/page_1.py")
    >>> if st.button("Page 2"):
    >>>     st.switch_page("pages/page_2.py")

    .. output ::
        https://doc-switch-page.streamlit.app/
        height: 350px

    