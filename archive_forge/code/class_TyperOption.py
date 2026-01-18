import errno
import inspect
import os
import sys
from enum import Enum
from gettext import gettext as _
from typing import (
import click
import click.core
import click.formatting
import click.parser
import click.shell_completion
import click.types
import click.utils
class TyperOption(click.core.Option):

    def __init__(self, *, param_decls: List[str], type: Optional[Union[click.types.ParamType, Any]]=None, required: Optional[bool]=None, default: Optional[Any]=None, callback: Optional[Callable[..., Any]]=None, nargs: Optional[int]=None, metavar: Optional[str]=None, expose_value: bool=True, is_eager: bool=False, envvar: Optional[Union[str, List[str]]]=None, shell_complete: Optional[Callable[[click.Context, click.Parameter, str], Union[List['click.shell_completion.CompletionItem'], List[str]]]]=None, autocompletion: Optional[Callable[..., Any]]=None, show_default: Union[bool, str]=False, prompt: Union[bool, str]=False, confirmation_prompt: Union[bool, str]=False, prompt_required: bool=True, hide_input: bool=False, is_flag: Optional[bool]=None, flag_value: Optional[Any]=None, multiple: bool=False, count: bool=False, allow_from_autoenv: bool=True, help: Optional[str]=None, hidden: bool=False, show_choices: bool=True, show_envvar: bool=False, rich_help_panel: Union[str, None]=None):
        super().__init__(param_decls=param_decls, type=type, required=required, default=default, callback=callback, nargs=nargs, metavar=metavar, expose_value=expose_value, is_eager=is_eager, envvar=envvar, show_default=show_default, prompt=prompt, confirmation_prompt=confirmation_prompt, hide_input=hide_input, is_flag=is_flag, flag_value=flag_value, multiple=multiple, count=count, allow_from_autoenv=allow_from_autoenv, help=help, hidden=hidden, show_choices=show_choices, show_envvar=show_envvar, prompt_required=prompt_required, shell_complete=shell_complete)
        _typer_param_setup_autocompletion_compat(self, autocompletion=autocompletion)
        self.rich_help_panel = rich_help_panel

    def _get_default_string(self, *, ctx: click.Context, show_default_is_str: bool, default_value: Union[List[Any], Tuple[Any, ...], str, Callable[..., Any], Any]) -> str:
        return _get_default_string(self, ctx=ctx, show_default_is_str=show_default_is_str, default_value=default_value)

    def _extract_default_help_str(self, *, ctx: click.Context) -> Optional[Union[Any, Callable[[], Any]]]:
        return _extract_default_help_str(self, ctx=ctx)

    def get_help_record(self, ctx: click.Context) -> Optional[Tuple[str, str]]:
        if self.hidden:
            return None
        any_prefix_is_slash = False

        def _write_opts(opts: Sequence[str]) -> str:
            nonlocal any_prefix_is_slash
            rv, any_slashes = click.formatting.join_options(opts)
            if any_slashes:
                any_prefix_is_slash = True
            if not self.is_flag and (not self.count):
                rv += f' {self.make_metavar()}'
            return rv
        rv = [_write_opts(self.opts)]
        if self.secondary_opts:
            rv.append(_write_opts(self.secondary_opts))
        help = self.help or ''
        extra = []
        if self.show_envvar:
            envvar = self.envvar
            if envvar is None:
                if self.allow_from_autoenv and ctx.auto_envvar_prefix is not None and (self.name is not None):
                    envvar = f'{ctx.auto_envvar_prefix}_{self.name.upper()}'
            if envvar is not None:
                var_str = envvar if isinstance(envvar, str) else ', '.join((str(d) for d in envvar))
                extra.append(_('env var: {var}').format(var=var_str))
        default_value = self._extract_default_help_str(ctx=ctx)
        show_default_is_str = isinstance(self.show_default, str)
        if show_default_is_str or (default_value is not None and (self.show_default or ctx.show_default)):
            default_string = self._get_default_string(ctx=ctx, show_default_is_str=show_default_is_str, default_value=default_value)
            if default_string:
                extra.append(_('default: {default}').format(default=default_string))
        if isinstance(self.type, click.types._NumberRangeBase):
            range_str = self.type._describe_range()
            if range_str:
                extra.append(range_str)
        if self.required:
            extra.append(_('required'))
        if extra:
            extra_str = '; '.join(extra)
            help = f'{help}  [{extra_str}]' if help else f'[{extra_str}]'
        return (('; ' if any_prefix_is_slash else ' / ').join(rv), help)