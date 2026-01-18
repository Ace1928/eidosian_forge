from __future__ import annotations
from enum import Enum
from types import ModuleType
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import ClassVar
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import util
from ..event import EventTarget
from ..pool import Pool
from ..pool import PoolProxiedConnection
from ..sql.compiler import Compiled as Compiled
from ..sql.compiler import Compiled  # noqa
from ..sql.compiler import TypeCompiler as TypeCompiler
from ..sql.compiler import TypeCompiler  # noqa
from ..util import immutabledict
from ..util.concurrency import await_only
from ..util.typing import Literal
from ..util.typing import NotRequired
from ..util.typing import Protocol
from ..util.typing import TypedDict
class CreateEnginePlugin:
    """A set of hooks intended to augment the construction of an
    :class:`_engine.Engine` object based on entrypoint names in a URL.

    The purpose of :class:`_engine.CreateEnginePlugin` is to allow third-party
    systems to apply engine, pool and dialect level event listeners without
    the need for the target application to be modified; instead, the plugin
    names can be added to the database URL.  Target applications for
    :class:`_engine.CreateEnginePlugin` include:

    * connection and SQL performance tools, e.g. which use events to track
      number of checkouts and/or time spent with statements

    * connectivity plugins such as proxies

    A rudimentary :class:`_engine.CreateEnginePlugin` that attaches a logger
    to an :class:`_engine.Engine` object might look like::


        import logging

        from sqlalchemy.engine import CreateEnginePlugin
        from sqlalchemy import event

        class LogCursorEventsPlugin(CreateEnginePlugin):
            def __init__(self, url, kwargs):
                # consume the parameter "log_cursor_logging_name" from the
                # URL query
                logging_name = url.query.get("log_cursor_logging_name", "log_cursor")

                self.log = logging.getLogger(logging_name)

            def update_url(self, url):
                "update the URL to one that no longer includes our parameters"
                return url.difference_update_query(["log_cursor_logging_name"])

            def engine_created(self, engine):
                "attach an event listener after the new Engine is constructed"
                event.listen(engine, "before_cursor_execute", self._log_event)


            def _log_event(
                self,
                conn,
                cursor,
                statement,
                parameters,
                context,
                executemany):

                self.log.info("Plugin logged cursor event: %s", statement)



    Plugins are registered using entry points in a similar way as that
    of dialects::

        entry_points={
            'sqlalchemy.plugins': [
                'log_cursor_plugin = myapp.plugins:LogCursorEventsPlugin'
            ]

    A plugin that uses the above names would be invoked from a database
    URL as in::

        from sqlalchemy import create_engine

        engine = create_engine(
            "mysql+pymysql://scott:tiger@localhost/test?"
            "plugin=log_cursor_plugin&log_cursor_logging_name=mylogger"
        )

    The ``plugin`` URL parameter supports multiple instances, so that a URL
    may specify multiple plugins; they are loaded in the order stated
    in the URL::

        engine = create_engine(
          "mysql+pymysql://scott:tiger@localhost/test?"
          "plugin=plugin_one&plugin=plugin_twp&plugin=plugin_three")

    The plugin names may also be passed directly to :func:`_sa.create_engine`
    using the :paramref:`_sa.create_engine.plugins` argument::

        engine = create_engine(
          "mysql+pymysql://scott:tiger@localhost/test",
          plugins=["myplugin"])

    .. versionadded:: 1.2.3  plugin names can also be specified
       to :func:`_sa.create_engine` as a list

    A plugin may consume plugin-specific arguments from the
    :class:`_engine.URL` object as well as the ``kwargs`` dictionary, which is
    the dictionary of arguments passed to the :func:`_sa.create_engine`
    call.  "Consuming" these arguments includes that they must be removed
    when the plugin initializes, so that the arguments are not passed along
    to the :class:`_engine.Dialect` constructor, where they will raise an
    :class:`_exc.ArgumentError` because they are not known by the dialect.

    As of version 1.4 of SQLAlchemy, arguments should continue to be consumed
    from the ``kwargs`` dictionary directly, by removing the values with a
    method such as ``dict.pop``. Arguments from the :class:`_engine.URL` object
    should be consumed by implementing the
    :meth:`_engine.CreateEnginePlugin.update_url` method, returning a new copy
    of the :class:`_engine.URL` with plugin-specific parameters removed::

        class MyPlugin(CreateEnginePlugin):
            def __init__(self, url, kwargs):
                self.my_argument_one = url.query['my_argument_one']
                self.my_argument_two = url.query['my_argument_two']
                self.my_argument_three = kwargs.pop('my_argument_three', None)

            def update_url(self, url):
                return url.difference_update_query(
                    ["my_argument_one", "my_argument_two"]
                )

    Arguments like those illustrated above would be consumed from a
    :func:`_sa.create_engine` call such as::

        from sqlalchemy import create_engine

        engine = create_engine(
          "mysql+pymysql://scott:tiger@localhost/test?"
          "plugin=myplugin&my_argument_one=foo&my_argument_two=bar",
          my_argument_three='bat'
        )

    .. versionchanged:: 1.4

        The :class:`_engine.URL` object is now immutable; a
        :class:`_engine.CreateEnginePlugin` that needs to alter the
        :class:`_engine.URL` should implement the newly added
        :meth:`_engine.CreateEnginePlugin.update_url` method, which
        is invoked after the plugin is constructed.

        For migration, construct the plugin in the following way, checking
        for the existence of the :meth:`_engine.CreateEnginePlugin.update_url`
        method to detect which version is running::

            class MyPlugin(CreateEnginePlugin):
                def __init__(self, url, kwargs):
                    if hasattr(CreateEnginePlugin, "update_url"):
                        # detect the 1.4 API
                        self.my_argument_one = url.query['my_argument_one']
                        self.my_argument_two = url.query['my_argument_two']
                    else:
                        # detect the 1.3 and earlier API - mutate the
                        # URL directly
                        self.my_argument_one = url.query.pop('my_argument_one')
                        self.my_argument_two = url.query.pop('my_argument_two')

                    self.my_argument_three = kwargs.pop('my_argument_three', None)

                def update_url(self, url):
                    # this method is only called in the 1.4 version
                    return url.difference_update_query(
                        ["my_argument_one", "my_argument_two"]
                    )

        .. seealso::

            :ref:`change_5526` - overview of the :class:`_engine.URL` change which
            also includes notes regarding :class:`_engine.CreateEnginePlugin`.


    When the engine creation process completes and produces the
    :class:`_engine.Engine` object, it is again passed to the plugin via the
    :meth:`_engine.CreateEnginePlugin.engine_created` hook.  In this hook, additional
    changes can be made to the engine, most typically involving setup of
    events (e.g. those defined in :ref:`core_event_toplevel`).

    """

    def __init__(self, url: URL, kwargs: Dict[str, Any]):
        """Construct a new :class:`.CreateEnginePlugin`.

        The plugin object is instantiated individually for each call
        to :func:`_sa.create_engine`.  A single :class:`_engine.
        Engine` will be
        passed to the :meth:`.CreateEnginePlugin.engine_created` method
        corresponding to this URL.

        :param url: the :class:`_engine.URL` object.  The plugin may inspect
         the :class:`_engine.URL` for arguments.  Arguments used by the
         plugin should be removed, by returning an updated :class:`_engine.URL`
         from the :meth:`_engine.CreateEnginePlugin.update_url` method.

         .. versionchanged::  1.4

            The :class:`_engine.URL` object is now immutable, so a
            :class:`_engine.CreateEnginePlugin` that needs to alter the
            :class:`_engine.URL` object should implement the
            :meth:`_engine.CreateEnginePlugin.update_url` method.

        :param kwargs: The keyword arguments passed to
         :func:`_sa.create_engine`.

        """
        self.url = url

    def update_url(self, url: URL) -> URL:
        """Update the :class:`_engine.URL`.

        A new :class:`_engine.URL` should be returned.   This method is
        typically used to consume configuration arguments from the
        :class:`_engine.URL` which must be removed, as they will not be
        recognized by the dialect.  The
        :meth:`_engine.URL.difference_update_query` method is available
        to remove these arguments.   See the docstring at
        :class:`_engine.CreateEnginePlugin` for an example.


        .. versionadded:: 1.4

        """
        raise NotImplementedError()

    def handle_dialect_kwargs(self, dialect_cls: Type[Dialect], dialect_args: Dict[str, Any]) -> None:
        """parse and modify dialect kwargs"""

    def handle_pool_kwargs(self, pool_cls: Type[Pool], pool_args: Dict[str, Any]) -> None:
        """parse and modify pool kwargs"""

    def engine_created(self, engine: Engine) -> None:
        """Receive the :class:`_engine.Engine`
        object when it is fully constructed.

        The plugin may make additional changes to the engine, such as
        registering engine or connection pool events.

        """