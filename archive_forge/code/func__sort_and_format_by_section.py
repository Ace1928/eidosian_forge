import logging
import sys
import textwrap
import warnings
import yaml
from oslo_config import cfg
from oslo_serialization import jsonutils
import stevedore
from oslo_policy import policy
def _sort_and_format_by_section(policies, output_format='yaml', include_help=True, exclude_deprecated=False):
    """Generate a list of policy section texts

    The text for a section will be created and returned one at a time. The
    sections are sorted first to provide for consistent output.

    Text is created in yaml format. This is done manually because PyYaml
    does not facilitate outputing comments.

    :param policies: A dict of {section1: [rule_default_1, rule_default_2],
                                section2: [rule_default_3]}
    :param output_format: The format of the file to output to.
    :param exclude_deprecated: If to exclude deprecated policy rule entries,
                               defaults to False.
    """
    for section in sorted(policies.keys()):
        rule_defaults = policies[section]
        for rule_default in rule_defaults:
            if output_format == 'yaml':
                yield _format_rule_default_yaml(rule_default, include_help=include_help, add_deprecated_rules=not exclude_deprecated)
            elif output_format == 'json':
                LOG.warning(policy.WARN_JSON)
                yield _format_rule_default_json(rule_default)