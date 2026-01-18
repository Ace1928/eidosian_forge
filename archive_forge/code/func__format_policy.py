import os
from docutils import nodes
from docutils.parsers import rst
from docutils.parsers.rst import directives
from docutils import statemachine
from oslo_config import cfg
from sphinx.util import logging
from sphinx.util.nodes import nested_parse_with_titles
from oslo_policy import generator
def _format_policy(namespaces):
    policies = generator.get_policies_dict(namespaces)
    for section in sorted(policies.keys()):
        for line in _format_policy_section(section, policies[section]):
            yield line