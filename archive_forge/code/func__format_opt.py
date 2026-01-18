from docutils import nodes
from docutils.parsers import rst
from docutils.parsers.rst import directives
from docutils.statemachine import ViewList
import oslo_i18n
from sphinx import addnodes
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain
from sphinx.domains import ObjType
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.nodes import make_refnode
from sphinx.util.nodes import nested_parse_with_titles
from oslo_config import cfg
from oslo_config import generator
def _format_opt(opt, group_name):
    opt_type = _TYPE_DESCRIPTIONS.get(type(opt), 'unknown type')
    yield ('.. oslo.config:option:: %s' % opt.dest)
    yield ''
    yield _indent(':Type: %s' % opt_type)
    for default in generator._format_defaults(opt):
        if default:
            yield _indent(':Default: ``%s``' % default)
        else:
            yield _indent(':Default: ``%r``' % default)
    if getattr(opt.type, 'min', None) is not None:
        yield _indent(':Minimum Value: %s' % opt.type.min)
    if getattr(opt.type, 'max', None) is not None:
        yield _indent(':Maximum Value: %s' % opt.type.max)
    if getattr(opt.type, 'choices', None):
        choices_text = ', '.join([_get_choice_text(choice) for choice in opt.type.choices])
        yield _indent(':Valid Values: %s' % choices_text)
    try:
        if opt.mutable:
            yield _indent(':Mutable: This option can be changed without restarting.')
    except AttributeError as err:
        import warnings
        if not isinstance(cfg.Opt, opt):
            warnings.warn('Incompatible option class for %s (%r): %s' % (opt.dest, opt.__class__, err))
        else:
            warnings.warn('Failed to fully format sample for %s: %s' % (opt.dest, err))
    if opt.advanced:
        yield _indent(':Advanced Option: Intended for advanced users and not used')
        yield _indent('by the majority of users, and might have a significant', 6)
        yield _indent('effect on stability and/or performance.', 6)
    if opt.sample_default:
        yield _indent('')
        yield _indent('This option has a sample default set, which means that')
        yield _indent('its actual default value may vary from the one documented')
        yield _indent('above.')
    try:
        help_text = opt.help % {'default': 'the value above'}
    except (TypeError, KeyError, ValueError):
        help_text = opt.help
    if help_text:
        yield ''
        for line in help_text.strip().splitlines():
            yield _indent(line.rstrip())
    if getattr(opt.type, 'choices', None) and (not all((x is None for x in opt.type.choices.values()))):
        yield ''
        yield _indent('.. rubric:: Possible values')
        for choice in opt.type.choices:
            yield ''
            yield _indent(_get_choice_text(choice))
            yield _indent(_indent(opt.type.choices[choice] or '<No description provided>'))
    if opt.deprecated_opts:
        yield ''
        for line in _list_table(['Group', 'Name'], ((d.group or group_name, d.name or opt.dest or 'UNSET') for d in opt.deprecated_opts), title='Deprecated Variations'):
            yield _indent(line)
    if opt.deprecated_for_removal:
        yield ''
        yield _indent('.. warning::')
        if opt.deprecated_since:
            yield _indent('   This option is deprecated for removal since %s.' % opt.deprecated_since)
        else:
            yield _indent('   This option is deprecated for removal.')
        yield _indent('   Its value may be silently ignored ')
        yield _indent('   in the future.')
        if opt.deprecated_reason:
            reason = ' '.join([x.strip() for x in opt.deprecated_reason.splitlines()])
            yield ''
            yield _indent('   :Reason: ' + reason)
    yield ''