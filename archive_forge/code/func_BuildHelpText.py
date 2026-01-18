from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.declarative import flags as declarative_config_flags
from googlecloudsdk.command_lib.util.declarative.clients import kcc_client
def BuildHelpText(singular, plural=None, service=None, begins_with_vowel=False):
    """Builds and returns help text for declarative export commands."""
    plural = plural or '{}s'.format(singular)
    singular_with_service = singular
    if service:
        singular_with_service = '{} {}'.format(service, singular)
    a_or_an = 'a'
    if begins_with_vowel:
        a_or_an = 'an'
    resource_name = '-'.join(singular.lower().split())
    detailed_help = {'brief': 'Export the configuration for {a_or_an} {singular_with_service}.'.format(a_or_an=a_or_an, singular_with_service=singular_with_service), 'DESCRIPTION': '            *{{command}}* exports the configuration for {a_or_an} {singular_with_service}.\n\n            {singular_capitalized} configurations can be exported in\n            Kubernetes Resource Model (krm) or Terraform HCL formats. The\n            default format is `krm`.\n\n            Specifying `--all` allows you to export the configurations for all\n            {plural} within the project.\n\n            Specifying `--path` allows you to export the configuration(s) to\n            a local directory.\n          '.format(singular_capitalized=singular.capitalize(), singular_with_service=singular_with_service, plural=plural, a_or_an=a_or_an), 'EXAMPLES': '            To export the configuration for {a_or_an} {singular}, run:\n\n              $ {{command}} my-{resource_name}\n\n            To export the configuration for {a_or_an} {singular} to a file, run:\n\n              $ {{command}} my-{resource_name} --path=/path/to/dir/\n\n            To export the configuration for {a_or_an} {singular} in Terraform\n            HCL format, run:\n\n              $ {{command}} my-{resource_name} --resource-format=terraform\n\n            To export the configurations for all {plural} within a\n            project, run:\n\n              $ {{command}} --all\n          '.format(singular=singular, plural=plural, resource_name=resource_name, a_or_an=a_or_an)}
    return detailed_help