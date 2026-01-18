import os
from docutils import nodes
from docutils.parsers import rst
from docutils.parsers.rst import directives
from docutils import statemachine
from oslo_config import cfg
from sphinx.util import logging
from sphinx.util.nodes import nested_parse_with_titles
from oslo_policy import generator
def _format_policy_rule(rule):
    """Output a definition list-style rule.

    For example::

        ``os_compute_api:servers:create``
            :Default: ``rule:admin_or_owner``
            :Operations:
              - **POST** ``/servers``

            Create a server
    """
    yield '``{}``'.format(rule.name)
    if rule.check_str:
        yield _indent(':Default: ``{}``'.format(rule.check_str))
    else:
        yield _indent(':Default: <empty string>')
    if hasattr(rule, 'operations'):
        yield _indent(':Operations:')
        for operation in rule.operations:
            yield _indent(_indent('- **{}** ``{}``'.format(operation['method'], operation['path'])))
    if hasattr(rule, 'scope_types') and rule.scope_types is not None:
        yield _indent(':Scope Types:')
        for scope_type in rule.scope_types:
            yield _indent(_indent('- **{}**'.format(scope_type)))
    yield ''
    if rule.description:
        for line in rule.description.strip().splitlines():
            yield _indent(line.rstrip())
    else:
        yield _indent('(no description provided)')
    yield ''