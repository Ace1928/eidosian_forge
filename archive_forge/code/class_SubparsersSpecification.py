from __future__ import annotations
import argparse
import dataclasses
from typing import (
from typing_extensions import Annotated, get_args, get_origin
from . import (
from ._typing import TypeForm
from .conf import _confstruct, _markers
@dataclasses.dataclass(frozen=True)
class SubparsersSpecification:
    """Structure for defining subparsers. Each subparser is a parser with a name."""
    name: str
    description: Optional[str]
    parser_from_name: Dict[str, ParserSpecification]
    intern_prefix: str
    required: bool
    default_instance: Any
    options: Tuple[Union[TypeForm[Any], Callable], ...]

    @staticmethod
    def from_field(field: _fields.FieldDefinition, type_from_typevar: Dict[TypeVar, TypeForm[Any]], parent_classes: Set[Type[Any]], intern_prefix: str, extern_prefix: str) -> Optional[SubparsersSpecification]:
        typ = _resolver.unwrap_annotated(field.type_or_callable)[0]
        if get_origin(typ) is not Union:
            return None
        options: List[Union[type, Callable]]
        options = [_resolver.apply_type_from_typevar(typ, type_from_typevar) for typ in get_args(typ)]
        options = [cast(Callable, none_proxy) if o is type(None) else o for o in options]
        for i, option in enumerate(options):
            _, found_subcommand_configs = _resolver.unwrap_annotated(option, _confstruct._SubcommandConfiguration)
            if len(found_subcommand_configs) > 0 and found_subcommand_configs[0].constructor_factory is not None:
                options[i] = Annotated.__class_getitem__((found_subcommand_configs[0].constructor_factory(), *_resolver.unwrap_annotated(option, Any)[1]))
        if not any([o is not none_proxy and _fields.is_nested_type(cast(type, o), _fields.MISSING_NONPROP) for o in options]):
            return None
        subcommand_config_from_name: Dict[str, _confstruct._SubcommandConfiguration] = {}
        subcommand_type_from_name: Dict[str, type] = {}
        for option in options:
            subcommand_name = _strings.subparser_name_from_type('' if _markers.OmitSubcommandPrefixes in field.markers else extern_prefix, type(None) if option is none_proxy else cast(type, option))
            option_unwrapped, found_subcommand_configs = _resolver.unwrap_annotated(option, _confstruct._SubcommandConfiguration)
            if len(found_subcommand_configs) != 0:
                assert len(found_subcommand_configs) == 1, f'Expected only one subcommand config, but {subcommand_name} has {len(found_subcommand_configs)}.'
                subcommand_config_from_name[subcommand_name] = found_subcommand_configs[0]
            subcommand_type_from_name[subcommand_name] = cast(type, option)
        if field.default is None or field.default in _fields.MISSING_SINGLETONS:
            default_name = None
        else:
            default_name = _subcommand_matching.match_subcommand(field.default, subcommand_config_from_name, subcommand_type_from_name)
            assert default_name is not None, f'`{extern_prefix}` was provided a default value of type {type(field.default)} but no matching subcommand was found. A type may be missing in the Union type declaration for `{extern_prefix}`, which currently expects {options}.'
        parser_from_name: Dict[str, ParserSpecification] = {}
        for option in options:
            subcommand_name = _strings.subparser_name_from_type('' if _markers.OmitSubcommandPrefixes in field.markers else extern_prefix, type(None) if option is none_proxy else cast(type, option))
            if subcommand_name in subcommand_config_from_name:
                subcommand_config = subcommand_config_from_name[subcommand_name]
            else:
                subcommand_config = _confstruct._SubcommandConfiguration('unused', description=None, default=_fields.MISSING_NONPROP, prefix_name=True, constructor_factory=None)
            if default_name == subcommand_name and (field.is_default_from_default_instance or subcommand_config.default in _fields.MISSING_SINGLETONS):
                subcommand_config = dataclasses.replace(subcommand_config, default=field.default)
            option_origin, annotations = _resolver.unwrap_annotated(option)
            annotations = tuple((a for a in annotations if not isinstance(a, _confstruct._SubcommandConfiguration)))
            if len(annotations) == 0:
                option = option_origin
            else:
                option = Annotated.__class_getitem__((option_origin,) + annotations)
            subparser = ParserSpecification.from_callable_or_type(Annotated.__class_getitem__((option,) + tuple(field.markers)) if len(field.markers) > 0 else option, description=subcommand_config.description, parent_classes=parent_classes, default_instance=subcommand_config.default, intern_prefix=intern_prefix, extern_prefix=extern_prefix, subcommand_prefix=intern_prefix, support_single_arg_types=True)
            subparser = dataclasses.replace(subparser, helptext_from_intern_prefixed_field_name={_strings.make_field_name([intern_prefix, k]): v for k, v in subparser.helptext_from_intern_prefixed_field_name.items()})
            parser_from_name[subcommand_name] = subparser
        required = field.default in _fields.MISSING_SINGLETONS
        if default_name is not None:
            default_parser = parser_from_name[default_name]
            if any(map(lambda arg: arg.lowered.required, default_parser.args)):
                required = True
            if default_parser.subparsers is not None and default_parser.subparsers.required:
                required = True
        if _markers.ConsolidateSubcommandArgs in field.markers:
            required = True
        description_parts = []
        if field.helptext is not None:
            description_parts.append(field.helptext)
        if not required and field.default not in _fields.MISSING_SINGLETONS:
            description_parts.append(f' (default: {default_name})')
        description = ' '.join(description_parts) if len(description_parts) > 0 else None
        return SubparsersSpecification(name=field.intern_name, description=description, parser_from_name=parser_from_name, intern_prefix=intern_prefix, required=required, default_instance=field.default, options=tuple(options))

    def apply(self, parent_parser: argparse.ArgumentParser) -> Tuple[argparse.ArgumentParser, ...]:
        title = 'subcommands'
        metavar = '{' + ','.join(self.parser_from_name.keys()) + '}'
        if not self.required:
            title = 'optional ' + title
            metavar = f'[{metavar}]'
        argparse_subparsers = parent_parser.add_subparsers(dest=_strings.make_subparser_dest(self.intern_prefix), description=self.description, required=self.required, title=title, metavar=metavar)
        subparser_tree_leaves: List[argparse.ArgumentParser] = []
        for name, subparser_def in self.parser_from_name.items():
            helptext = subparser_def.description.replace('%', '%%')
            if len(helptext) > 0:
                helptext = _arguments._rich_tag_if_enabled(helptext.strip(), 'helptext')
            subparser = argparse_subparsers.add_parser(name, formatter_class=_argparse_formatter.TyroArgparseHelpFormatter, help=helptext, allow_abbrev=False)
            assert isinstance(subparser, _argparse_formatter.TyroArgumentParser)
            assert isinstance(parent_parser, _argparse_formatter.TyroArgumentParser)
            subparser._parsing_known_args = parent_parser._parsing_known_args
            subparser._parser_specification = parent_parser._parser_specification
            subparser._args = parent_parser._args
            subparser_tree_leaves.extend(subparser_def.apply(subparser))
        return tuple(subparser_tree_leaves)