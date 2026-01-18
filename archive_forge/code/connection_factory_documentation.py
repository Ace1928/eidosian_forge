from __future__ import annotations
import os
import re
from datetime import timedelta
from typing import Any, Final, Literal, TypeVar, overload
from streamlit.connections import (
from streamlit.deprecation_util import deprecate_obj_name
from streamlit.errors import StreamlitAPIException
from streamlit.runtime.caching import cache_resource
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.secrets import secrets_singleton
Create a new connection to a data store or API, or return an existing one.

    Config options, credentials, secrets, etc. for connections are taken from various
    sources:

    - Any connection-specific configuration files.
    - An app's ``secrets.toml`` files.
    - The kwargs passed to this function.

    Parameters
    ----------
    name : str
        The connection name used for secrets lookup in ``[connections.<name>]``.
        Type will be inferred from passing ``"sql"``, ``"snowflake"``, or ``"snowpark"``.
    type : str, connection class, or None
        The type of connection to create. It can be a keyword (``"sql"``, ``"snowflake"``,
        or ``"snowpark"``), a path to an importable class, or an imported class reference.
        All classes must extend ``st.connections.BaseConnection`` and implement the
        ``_connect()`` method. If the type kwarg is None, a ``type`` field must be set in
        the connection's section in ``secrets.toml``.
    max_entries : int or None
        The maximum number of connections to keep in the cache, or None
        for an unbounded cache. (When a new entry is added to a full cache,
        the oldest cached entry will be removed.) The default is None.
    ttl : float, timedelta, or None
        The maximum number of seconds to keep results in the cache, or
        None if cached results should not expire. The default is None.
    **kwargs : any
        Additional connection specific kwargs that are passed to the Connection's
        ``_connect()`` method. Learn more from the specific Connection's documentation.

    Returns
    -------
    Connection object
        An initialized Connection object of the specified type.

    Examples
    --------
    The easiest way to create a first-party (SQL, Snowflake, or Snowpark) connection is
    to use their default names and define corresponding sections in your ``secrets.toml``
    file.

    >>> import streamlit as st
    >>> conn = st.connection("sql") # Config section defined in [connections.sql] in secrets.toml.

    Creating a SQLConnection with a custom name requires you to explicitly specify the
    type. If type is not passed as a kwarg, it must be set in the appropriate section of
    ``secrets.toml``.

    >>> import streamlit as st
    >>> conn1 = st.connection("my_sql_connection", type="sql") # Config section defined in [connections.my_sql_connection].
    >>> conn2 = st.connection("my_other_sql_connection") # type must be set in [connections.my_other_sql_connection].

    Passing the full module path to the connection class that you want to use can be
    useful, especially when working with a custom connection:

    >>> import streamlit as st
    >>> conn = st.connection("my_sql_connection", type="streamlit.connections.SQLConnection")

    Finally, you can pass the connection class to use directly to this function. Doing
    so allows static type checking tools such as ``mypy`` to infer the exact return
    type of ``st.connection``.

    >>> import streamlit as st
    >>> from streamlit.connections import SQLConnection
    >>> conn = st.connection("my_sql_connection", type=SQLConnection)
    