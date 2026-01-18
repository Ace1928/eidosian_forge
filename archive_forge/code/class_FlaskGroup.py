from __future__ import annotations
import ast
import collections.abc as cabc
import importlib.metadata
import inspect
import os
import platform
import re
import sys
import traceback
import typing as t
from functools import update_wrapper
from operator import itemgetter
from types import ModuleType
import click
from click.core import ParameterSource
from werkzeug import run_simple
from werkzeug.serving import is_running_from_reloader
from werkzeug.utils import import_string
from .globals import current_app
from .helpers import get_debug_flag
from .helpers import get_load_dotenv
class FlaskGroup(AppGroup):
    """Special subclass of the :class:`AppGroup` group that supports
    loading more commands from the configured Flask app.  Normally a
    developer does not have to interface with this class but there are
    some very advanced use cases for which it makes sense to create an
    instance of this. see :ref:`custom-scripts`.

    :param add_default_commands: if this is True then the default run and
        shell commands will be added.
    :param add_version_option: adds the ``--version`` option.
    :param create_app: an optional callback that is passed the script info and
        returns the loaded app.
    :param load_dotenv: Load the nearest :file:`.env` and :file:`.flaskenv`
        files to set environment variables. Will also change the working
        directory to the directory containing the first file found.
    :param set_debug_flag: Set the app's debug flag.

    .. versionchanged:: 2.2
        Added the ``-A/--app``, ``--debug/--no-debug``, ``-e/--env-file`` options.

    .. versionchanged:: 2.2
        An app context is pushed when running ``app.cli`` commands, so
        ``@with_appcontext`` is no longer required for those commands.

    .. versionchanged:: 1.0
        If installed, python-dotenv will be used to load environment variables
        from :file:`.env` and :file:`.flaskenv` files.
    """

    def __init__(self, add_default_commands: bool=True, create_app: t.Callable[..., Flask] | None=None, add_version_option: bool=True, load_dotenv: bool=True, set_debug_flag: bool=True, **extra: t.Any) -> None:
        params = list(extra.pop('params', None) or ())
        params.extend((_env_file_option, _app_option, _debug_option))
        if add_version_option:
            params.append(version_option)
        if 'context_settings' not in extra:
            extra['context_settings'] = {}
        extra['context_settings'].setdefault('auto_envvar_prefix', 'FLASK')
        super().__init__(params=params, **extra)
        self.create_app = create_app
        self.load_dotenv = load_dotenv
        self.set_debug_flag = set_debug_flag
        if add_default_commands:
            self.add_command(run_command)
            self.add_command(shell_command)
            self.add_command(routes_command)
        self._loaded_plugin_commands = False

    def _load_plugin_commands(self) -> None:
        if self._loaded_plugin_commands:
            return
        if sys.version_info >= (3, 10):
            from importlib import metadata
        else:
            import importlib_metadata as metadata
        for ep in metadata.entry_points(group='flask.commands'):
            self.add_command(ep.load(), ep.name)
        self._loaded_plugin_commands = True

    def get_command(self, ctx: click.Context, name: str) -> click.Command | None:
        self._load_plugin_commands()
        rv = super().get_command(ctx, name)
        if rv is not None:
            return rv
        info = ctx.ensure_object(ScriptInfo)
        try:
            app = info.load_app()
        except NoAppException as e:
            click.secho(f'Error: {e.format_message()}\n', err=True, fg='red')
            return None
        if not current_app or current_app._get_current_object() is not app:
            ctx.with_resource(app.app_context())
        return app.cli.get_command(ctx, name)

    def list_commands(self, ctx: click.Context) -> list[str]:
        self._load_plugin_commands()
        rv = set(super().list_commands(ctx))
        info = ctx.ensure_object(ScriptInfo)
        try:
            rv.update(info.load_app().cli.list_commands(ctx))
        except NoAppException as e:
            click.secho(f'Error: {e.format_message()}\n', err=True, fg='red')
        except Exception:
            click.secho(f'{traceback.format_exc()}\n', err=True, fg='red')
        return sorted(rv)

    def make_context(self, info_name: str | None, args: list[str], parent: click.Context | None=None, **extra: t.Any) -> click.Context:
        os.environ['FLASK_RUN_FROM_CLI'] = 'true'
        if get_load_dotenv(self.load_dotenv):
            load_dotenv()
        if 'obj' not in extra and 'obj' not in self.context_settings:
            extra['obj'] = ScriptInfo(create_app=self.create_app, set_debug_flag=self.set_debug_flag)
        return super().make_context(info_name, args, parent=parent, **extra)

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if not args and self.no_args_is_help:
            _env_file_option.handle_parse_result(ctx, {}, [])
            _app_option.handle_parse_result(ctx, {}, [])
        return super().parse_args(ctx, args)