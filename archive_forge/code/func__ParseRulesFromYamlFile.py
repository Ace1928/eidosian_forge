from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.networks.subnets import flags as subnet_flags
from googlecloudsdk.command_lib.compute.routers.nats import flags as nat_flags
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
import six
def _ParseRulesFromYamlFile(file_path, compute_holder):
    """Parses NAT Rules from the given YAML file."""
    with files.FileReader(file_path) as import_file:
        rules_yaml = yaml.load(import_file)
        if 'rules' not in rules_yaml:
            raise calliope_exceptions.InvalidArgumentException('--rules', "The YAML file must contain the 'rules' attribute")
        return [_CreateRule(rule_yaml, compute_holder) for rule_yaml in rules_yaml['rules']]