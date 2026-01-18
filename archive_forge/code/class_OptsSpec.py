from itertools import groupby
import numpy as np
import param
import pyparsing as pp
from ..core.options import Cycle, Options, Palette
from ..core.util import merge_option_dicts
from ..operation import Compositor
from .transform import dim
class OptsSpec(Parser):
    """
    An OptsSpec is a string specification that describes an
    OptionTree. It is a list of tree path specifications (using dotted
    syntax) separated by keyword lists for any of the style, plotting
    or normalization options. These keyword lists are denoted
    'plot(..)', 'style(...)' and 'norm(...)'  respectively.  These
    three groups may be specified even more concisely using keyword
    lists delimited by square brackets, parentheses and braces
    respectively.  All these sets are optional and may be supplied in
    any order.

    For instance, the following string:

    Image (interpolation=None) plot(show_title=False) Curve style(color='r')

    Would specify an OptionTree where Image has "interpolation=None"
    for style and 'show_title=False' for plot options. The Curve has a
    style set such that color='r'.

    The parser is fairly forgiving; commas between keywords are
    optional and additional spaces are often allowed. The only
    restriction is that keywords *must* be immediately followed by the
    '=' sign (no space).
    """
    plot_options_short = pp.nestedExpr('[', ']', content=pp.OneOrMore(pp.Word(allowed) ^ pp.quotedString)).setResultsName('plot_options')
    plot_options_long = pp.nestedExpr(opener='plot[', closer=']', content=pp.OneOrMore(pp.Word(allowed) ^ pp.quotedString)).setResultsName('plot_options')
    plot_options = plot_options_short | plot_options_long
    style_options_short = pp.nestedExpr(opener='(', closer=')', ignoreExpr=None).setResultsName('style_options')
    style_options_long = pp.nestedExpr(opener='style(', closer=')', ignoreExpr=None).setResultsName('style_options')
    style_options = style_options_short | style_options_long
    norm_options_short = pp.nestedExpr(opener='{', closer='}', ignoreExpr=None).setResultsName('norm_options')
    norm_options_long = pp.nestedExpr(opener='norm{', closer='}', ignoreExpr=None).setResultsName('norm_options')
    norm_options = norm_options_short | norm_options_long
    compositor_ops = pp.MatchFirst([pp.Literal(el.group) for el in Compositor.definitions if el.group])
    dotted_path = pp.Combine(pp.Word(ascii_uppercase, exact=1) + pp.Word(pp.alphanums + '._'))
    pathspec = (dotted_path | compositor_ops).setResultsName('pathspec')
    spec_group = pp.Group(pathspec + (pp.Optional(norm_options) & pp.Optional(plot_options) & pp.Optional(style_options)))
    opts_spec = pp.OneOrMore(spec_group)
    aliases = {'horizontal_spacing': 'hspace', 'vertical_spacing': 'vspace', 'figure_alpha': '    fig_alpha', 'figure_bounds': 'fig_bounds', 'figure_inches': 'fig_inches', 'figure_latex': 'fig_latex', 'figure_rcparams': 'fig_rcparams', 'figure_size': 'fig_size', 'show_xaxis': 'xaxis', 'show_yaxis': 'yaxis'}
    deprecations = []

    @classmethod
    def process_normalization(cls, parse_group):
        """
        Given a normalization parse group (i.e. the contents of the
        braces), validate the option list and compute the appropriate
        integer value for the normalization plotting option.
        """
        if 'norm_options' not in parse_group:
            return None
        opts = parse_group['norm_options'][0].asList()
        if opts == []:
            return None
        options = ['+framewise', '-framewise', '+axiswise', '-axiswise']
        for normopt in options:
            if opts.count(normopt) > 1:
                raise SyntaxError('Normalization specification must not contain repeated %r' % normopt)
        if not all((opt in options for opt in opts)):
            raise SyntaxError(f'Normalization option not one of {', '.join(options)}')
        excluded = [('+framewise', '-framewise'), ('+axiswise', '-axiswise')]
        for pair in excluded:
            if all((exclude in opts for exclude in pair)):
                raise SyntaxError(f'Normalization specification cannot contain both {pair[0]} and {pair[1]}')
        if len(opts) == 1 and opts[0].endswith('framewise'):
            axiswise = False
            framewise = True if '+framewise' in opts else False
        elif len(opts) == 1 and opts[0].endswith('axiswise'):
            framewise = False
            axiswise = True if '+axiswise' in opts else False
        else:
            axiswise = True if '+axiswise' in opts else False
            framewise = True if '+framewise' in opts else False
        return dict(axiswise=axiswise, framewise=framewise)

    @classmethod
    def _group_paths_without_options(cls, line_parse_result):
        """
        Given a parsed options specification as a list of groups, combine
        groups without options with the first subsequent group which has
        options.
        A line of the form
            'A B C [opts] D E [opts_2]'
        results in
            [({A, B, C}, [opts]), ({D, E}, [opts_2])]
        """
        active_pathspecs = set()
        for group in line_parse_result:
            active_pathspecs.add(group['pathspec'])
            has_options = 'norm_options' in group or 'plot_options' in group or 'style_options' in group
            if has_options:
                yield (active_pathspecs, group)
                active_pathspecs = set()
        if active_pathspecs:
            yield (active_pathspecs, {})

    @classmethod
    def apply_deprecations(cls, path):
        """Convert any potentially deprecated paths and issue appropriate warnings"""
        split = path.split('.')
        msg = 'Element {old} deprecated. Use {new} instead.'
        for old, new in cls.deprecations:
            if split[0] == old:
                parsewarning.warning(msg.format(old=old, new=new))
                return '.'.join([new] + split[1:])
        return path

    @classmethod
    def parse(cls, line, ns=None):
        """
        Parse an options specification, returning a dictionary with
        path keys and {'plot':<options>, 'style':<options>} values.
        """
        if ns is None:
            ns = {}
        parses = [p for p in cls.opts_spec.scanString(line)]
        if len(parses) != 1:
            raise SyntaxError('Invalid specification syntax.')
        else:
            e = parses[0][2]
            processed = line[:e]
            if processed.strip() != line.strip():
                raise SyntaxError(f'Failed to parse remainder of string: {line[e:]!r}')
        grouped_paths = cls._group_paths_without_options(cls.opts_spec.parseString(line))
        parse = {}
        for pathspecs, group in grouped_paths:
            options = {}
            normalization = cls.process_normalization(group)
            if normalization is not None:
                options['norm'] = normalization
            if 'plot_options' in group:
                plotopts = group['plot_options'][0]
                opts = cls.todict(plotopts, 'brackets', ns=ns)
                options['plot'] = {cls.aliases.get(k, k): v for k, v in opts.items()}
            if 'style_options' in group:
                styleopts = group['style_options'][0]
                opts = cls.todict(styleopts, 'parens', ns=ns)
                options['style'] = {cls.aliases.get(k, k): v for k, v in opts.items()}
            for pathspec in pathspecs:
                parse[pathspec] = merge_option_dicts(parse.get(pathspec, {}), options)
        return {cls.apply_deprecations(path): {option_type: Options(**option_pairs) for option_type, option_pairs in options.items()} for path, options in parse.items()}

    @classmethod
    def parse_options(cls, line, ns=None):
        """
        Similar to parse but returns a list of Options objects instead
        of the dictionary format.
        """
        if ns is None:
            ns = {}
        parsed = cls.parse(line, ns=ns)
        options_list = []
        for spec in sorted(parsed.keys()):
            options = parsed[spec]
            merged = {}
            for group in options.values():
                merged = dict(group.kwargs, **merged)
            options_list.append(Options(spec, **merged))
        return options_list