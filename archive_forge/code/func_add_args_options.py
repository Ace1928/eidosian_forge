import re
import click
import json
from .instance import import_module
from ..interfaces.base import InputMultiPath, traits
from ..interfaces.base.support import get_trait_desc
def add_args_options(arg_parser, interface):
    """Add arguments to `arg_parser` to create a CLI for `interface`."""
    inputs = interface.input_spec()
    for name, spec in sorted(interface.inputs.traits(transient=None).items()):
        desc = '\n'.join(get_trait_desc(inputs, name, spec))[len(name) + 2:]
        desc = desc.replace('%', '%%')
        args = {}
        has_multiple_inner_traits = False
        if spec.is_trait_type(traits.Bool):
            args['default'] = getattr(inputs, name)
            args['action'] = 'store_true'
        if not spec.inner_traits:
            if not spec.is_trait_type(traits.TraitCompound):
                trait_type = type(spec.trait_type.default_value)
            if trait_type in (bytes, str, int, float):
                if trait_type == bytes:
                    trait_type = str
                args['type'] = trait_type
        elif len(spec.inner_traits) == 1:
            trait_type = type(spec.inner_traits[0].trait_type.default_value)
            if trait_type == bytes:
                trait_type = str
            if trait_type in (bytes, bool, str, int, float):
                args['type'] = trait_type
        elif len(spec.inner_traits) > 1:
            if not spec.is_trait_type(traits.Dict):
                has_multiple_inner_traits = True
        if getattr(spec, 'mandatory', False):
            if spec.is_trait_type(InputMultiPath):
                args['nargs'] = '+'
            elif spec.is_trait_type(traits.List):
                if spec.trait_type.minlen == spec.trait_type.maxlen and spec.trait_type.maxlen:
                    args['nargs'] = spec.trait_type.maxlen
                else:
                    args['nargs'] = '+'
            elif spec.is_trait_type(traits.Dict):
                args['type'] = json.loads
            if has_multiple_inner_traits:
                raise NotImplementedError('This interface cannot be used. via the command line as multiple inner traits are currently not supported for mandatory argument: {}.'.format(name))
            arg_parser.add_argument(name, help=desc, **args)
        else:
            if spec.is_trait_type(InputMultiPath):
                args['nargs'] = '*'
            elif spec.is_trait_type(traits.List):
                if spec.trait_type.minlen == spec.trait_type.maxlen and spec.trait_type.maxlen:
                    args['nargs'] = spec.trait_type.maxlen
                else:
                    args['nargs'] = '*'
            if not has_multiple_inner_traits:
                arg_parser.add_argument('--%s' % name, dest=name, help=desc, **args)
    return arg_parser