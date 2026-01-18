import inspect
import io
from typing import (
import click
import click.shell_completion
class OptionInfo(ParameterInfo):

    def __init__(self, *, default: Optional[Any]=None, param_decls: Optional[Sequence[str]]=None, callback: Optional[Callable[..., Any]]=None, metavar: Optional[str]=None, expose_value: bool=True, is_eager: bool=False, envvar: Optional[Union[str, List[str]]]=None, shell_complete: Optional[Callable[[click.Context, click.Parameter, str], Union[List['click.shell_completion.CompletionItem'], List[str]]]]=None, autocompletion: Optional[Callable[..., Any]]=None, default_factory: Optional[Callable[[], Any]]=None, parser: Optional[Callable[[str], Any]]=None, click_type: Optional[click.ParamType]=None, show_default: Union[bool, str]=True, prompt: Union[bool, str]=False, confirmation_prompt: bool=False, prompt_required: bool=True, hide_input: bool=False, is_flag: Optional[bool]=None, flag_value: Optional[Any]=None, count: bool=False, allow_from_autoenv: bool=True, help: Optional[str]=None, hidden: bool=False, show_choices: bool=True, show_envvar: bool=True, case_sensitive: bool=True, min: Optional[Union[int, float]]=None, max: Optional[Union[int, float]]=None, clamp: bool=False, formats: Optional[List[str]]=None, mode: Optional[str]=None, encoding: Optional[str]=None, errors: Optional[str]='strict', lazy: Optional[bool]=None, atomic: bool=False, exists: bool=False, file_okay: bool=True, dir_okay: bool=True, writable: bool=False, readable: bool=True, resolve_path: bool=False, allow_dash: bool=False, path_type: Union[None, Type[str], Type[bytes]]=None, rich_help_panel: Union[str, None]=None):
        super().__init__(default=default, param_decls=param_decls, callback=callback, metavar=metavar, expose_value=expose_value, is_eager=is_eager, envvar=envvar, shell_complete=shell_complete, autocompletion=autocompletion, default_factory=default_factory, parser=parser, click_type=click_type, show_default=show_default, show_choices=show_choices, show_envvar=show_envvar, help=help, hidden=hidden, case_sensitive=case_sensitive, min=min, max=max, clamp=clamp, formats=formats, mode=mode, encoding=encoding, errors=errors, lazy=lazy, atomic=atomic, exists=exists, file_okay=file_okay, dir_okay=dir_okay, writable=writable, readable=readable, resolve_path=resolve_path, allow_dash=allow_dash, path_type=path_type, rich_help_panel=rich_help_panel)
        self.prompt = prompt
        self.confirmation_prompt = confirmation_prompt
        self.prompt_required = prompt_required
        self.hide_input = hide_input
        self.is_flag = is_flag
        self.flag_value = flag_value
        self.count = count
        self.allow_from_autoenv = allow_from_autoenv