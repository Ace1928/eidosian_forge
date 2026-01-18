from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.command_lib.kuberun import pretty_print
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.core.resource import resource_printer
from six.moves.urllib_parse import urlparse
class ComponentStatusPrinter(cp.CustomPrinterBase):
    """Prints the KubeRun Component Status in a custom human-readable format."""

    @staticmethod
    def Register(parser):
        """Register this custom printer with the given parser."""
        resource_printer.RegisterFormatter(_PRINTER_FORMAT, ComponentStatusPrinter, hidden=True)
        parser.display_info.AddFormat(_PRINTER_FORMAT)

    def Transform(self, record):
        """Transform ComponentStatus into the output structure of marker classes.

    Args:
      record: a dict object

    Returns:
      lines formatted for output
    """
        con = console_attr.GetConsoleAttr()
        component = record['status']
        status = con.Colorize(pretty_print.GetReadySymbol(component.deployment_state), pretty_print.GetReadyColor(component.deployment_state))
        component_url = urlparse(component.url)
        results = [cp.Section([con.Emphasize('{} Component {} in environment {}'.format(status, component.name, record['environment'])), 'Deployed at {} from commit {}\n'.format(component.deployment_time, component.commit_id)]), cp.Section([cp.Labeled([('Component Service(s)', cp.Lines(component.services))])]), cp.Section(['\nGet more details about services using kuberun core services describe SERVICE'])]
        if component.deployment_state == 'Ready':
            results.append(cp.Section(['\nTo invoke this component, run:', '  curl {}'.format(component.url), 'OR', '  curl -H "Host: {}" {}://{}'.format(component_url.netloc, component_url.scheme, record['ingressIp'])]))
        elif component.deployment_state == 'Failed':
            msg = '\n! Deployment failed with message: {}'.format(component.deployment_message)
            results.append(con.Emphasize(con.Colorize(msg, 'yellow')))
        return cp.Lines(results)