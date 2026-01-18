from itertools import groupby
import numpy as np
import param
import pyparsing as pp
from ..core.options import Cycle, Options, Palette
from ..core.util import merge_option_dicts
from ..operation import Compositor
from .transform import dim
class CompositorSpec(Parser):
    """
    The syntax for defining a set of compositor is as follows:

    [ mode op(spec) [settings] value ]+

    The components are:

    mode      : Operation mode, either 'data' or 'display'.
    group     : Value identifier with capitalized initial letter.
    op        : The name of the operation to apply.
    spec      : Overlay specification of form (A * B) where A and B are
                 dotted path specifications.
    settings  : Optional list of keyword arguments to be used as
                parameters to the operation (in square brackets).
    """
    mode = pp.Word(pp.alphas + pp.nums + '_').setResultsName('mode')
    op = pp.Word(pp.alphas + pp.nums + '_').setResultsName('op')
    overlay_spec = pp.nestedExpr(opener='(', closer=')', ignoreExpr=None).setResultsName('spec')
    value = pp.Word(pp.alphas + pp.nums + '_').setResultsName('value')
    op_settings = pp.nestedExpr(opener='[', closer=']', ignoreExpr=None).setResultsName('op_settings')
    compositor_spec = pp.OneOrMore(pp.Group(mode + op + overlay_spec + value + pp.Optional(op_settings)))

    @classmethod
    def parse(cls, line, ns=None):
        """
        Parse compositor specifications, returning a list Compositors
        """
        if ns is None:
            ns = {}
        definitions = []
        parses = [p for p in cls.compositor_spec.scanString(line)]
        if len(parses) != 1:
            raise SyntaxError('Invalid specification syntax.')
        else:
            e = parses[0][2]
            processed = line[:e]
            if processed.strip() != line.strip():
                raise SyntaxError(f'Failed to parse remainder of string: {line[e:]!r}')
        opmap = {op.__name__: op for op in Compositor.operations}
        for group in cls.compositor_spec.parseString(line):
            if 'mode' not in group or group['mode'] not in ['data', 'display']:
                raise SyntaxError('Either data or display mode must be specified.')
            mode = group['mode']
            kwargs = {}
            operation = opmap[group['op']]
            spec = ' '.join(group['spec'].asList()[0])
            if group['op'] not in opmap:
                raise SyntaxError('Operation %s not available for use with compositors.' % group['op'])
            if 'op_settings' in group:
                kwargs = cls.todict(group['op_settings'][0], 'brackets', ns=ns)
            definition = Compositor(str(spec), operation, str(group['value']), mode, **kwargs)
            definitions.append(definition)
        return definitions