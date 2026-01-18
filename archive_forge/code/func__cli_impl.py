import argparse
import dataclasses
import pathlib
import sys
import warnings
from typing import (
import shtab
from typing_extensions import Literal
from . import (
from ._typing import TypeForm
def _cli_impl(f: Union[TypeForm[OutT], Callable[..., OutT]], *, prog: Optional[str]=None, description: Optional[str], args: Optional[Sequence[str]], default: Optional[OutT], return_parser: bool, return_unknown_args: bool, **deprecated_kwargs) -> Union[OutT, argparse.ArgumentParser, Tuple[OutT, List[str]]]:
    """Helper for stitching the `tyro` pipeline together."""
    if 'default_instance' in deprecated_kwargs:
        warnings.warn('`default_instance=` is deprecated! use `default=` instead.', stacklevel=2)
        default = deprecated_kwargs['default_instance']
    if deprecated_kwargs.get('avoid_subparsers', False):
        f = conf.AvoidSubcommands[f]
        warnings.warn('`avoid_subparsers=` is deprecated! use `tyro.conf.AvoidSubparsers[]` instead.', stacklevel=2)
    default_instance_internal: Union[_fields.NonpropagatingMissingType, OutT] = _fields.MISSING_NONPROP if default is None else default
    if not _fields.is_nested_type(cast(type, f), default_instance_internal):
        dummy_field = cast(dataclasses.Field, dataclasses.field())
        f = dataclasses.make_dataclass(cls_name='dummy', fields=[(_strings.dummy_field_name, cast(type, f), dummy_field)], frozen=True)
        default_instance_internal = f(default_instance_internal)
        dummy_wrapped = True
    else:
        dummy_wrapped = False
    args = list(sys.argv[1:]) if args is None else list(args)
    modified_args: Dict[str, str] = {}
    for index, arg in enumerate(args):
        if not arg.startswith('--'):
            continue
        delimeter = _strings.get_delimeter()
        to_swap_delimeter = '-' if delimeter == '_' else '_'
        if '=' in arg:
            arg, _, val = arg.partition('=')
            fixed = '--' + arg[2:].replace(to_swap_delimeter, delimeter) + '=' + val
        else:
            fixed = '--' + arg[2:].replace(to_swap_delimeter, delimeter)
        if return_unknown_args and fixed in modified_args and (modified_args[fixed] != arg):
            raise RuntimeError('Ambiguous arguments: ' + modified_args[fixed] + ' and ' + arg)
        modified_args[fixed] = arg
        args[index] = fixed
    print_completion = len(args) >= 2 and args[0] == '--tyro-print-completion'
    write_completion = len(args) >= 3 and args[0] == '--tyro-write-completion'
    completion_shell = None
    completion_target_path = None
    if print_completion or write_completion:
        completion_shell = args[1]
    if write_completion:
        completion_target_path = pathlib.Path(args[2])
    if print_completion or write_completion or return_parser:
        _arguments.USE_RICH = False
    else:
        _arguments.USE_RICH = True
    parser_spec = _parsers.ParserSpecification.from_callable_or_type(f, description=description, parent_classes=set(), default_instance=default_instance_internal, intern_prefix='', extern_prefix='', subcommand_prefix='')
    with _argparse_formatter.ansi_context():
        parser = _argparse_formatter.TyroArgumentParser(prog=prog, formatter_class=_argparse_formatter.TyroArgparseHelpFormatter, allow_abbrev=False)
        parser._parser_specification = parser_spec
        parser._parsing_known_args = return_unknown_args
        parser._args = args
        parser_spec.apply(parser)
        if return_parser:
            _arguments.USE_RICH = True
            return parser
        if print_completion or write_completion:
            _arguments.USE_RICH = True
            assert completion_shell in ('bash', 'zsh', 'tcsh'), f'Shell should be one `bash`, `zsh`, or `tcsh`, but got {completion_shell}'
            if write_completion:
                assert completion_target_path is not None
                completion_target_path.write_text(shtab.complete(parser=parser, shell=completion_shell, root_prefix=f'tyro_{parser.prog}'))
            else:
                print(shtab.complete(parser=parser, shell=completion_shell, root_prefix=f'tyro_{parser.prog}'))
            sys.exit()
        if return_unknown_args:
            namespace, unknown_args = parser.parse_known_args(args=args)
        else:
            unknown_args = None
            namespace = parser.parse_args(args=args)
        value_from_prefixed_field_name = vars(namespace)
    if dummy_wrapped:
        value_from_prefixed_field_name = {k.replace(_strings.dummy_field_name, ''): v for k, v in value_from_prefixed_field_name.items()}
    try:
        out, consumed_keywords = _calling.call_from_args(f, parser_spec, default_instance_internal, value_from_prefixed_field_name, field_name_prefix='')
    except _calling.InstantiationError as e:
        assert isinstance(e, _calling.InstantiationError)
        from rich.console import Console, Group, RenderableType
        from rich.padding import Padding
        from rich.panel import Panel
        from rich.rule import Rule
        from rich.style import Style
        from ._argparse_formatter import THEME
        console = Console(theme=THEME.as_rich_theme())
        console.print(Panel(Group(f'[bright_red][bold]Error parsing {(e.arg.lowered.name_or_flag if isinstance(e.arg, _arguments.ArgumentDefinition) else e.arg)}[/bold]:[/bright_red] {e.message}', *cast(List[RenderableType], [] if not isinstance(e.arg, _arguments.ArgumentDefinition) or e.arg.lowered.help is None else [Rule(style=Style(color='red')), 'Argument helptext:', Padding(Group(f'{e.arg.lowered.name_or_flag} [bold]{e.arg.lowered.metavar}[/bold]', e.arg.lowered.help), pad=(0, 0, 0, 4)), Rule(style=Style(color='red')), f'For full helptext, see [bold]{parser.prog} --help[/bold]'])), title='[bold]Value error[/bold]', title_align='left', border_style=Style(color='red')))
        sys.exit(2)
    assert len(value_from_prefixed_field_name.keys() - consumed_keywords) == 0, f'Parsed {value_from_prefixed_field_name.keys()}, but only consumed {consumed_keywords}'
    if dummy_wrapped:
        out = getattr(out, _strings.dummy_field_name)
    if return_unknown_args:
        assert unknown_args is not None, 'Should have parsed with `parse_known_args()`'
        unknown_args = [modified_args.get(arg, arg) for arg in unknown_args]
        return (out, unknown_args)
    else:
        assert unknown_args is None, 'Should have parsed with `parse_args()`'
        return out