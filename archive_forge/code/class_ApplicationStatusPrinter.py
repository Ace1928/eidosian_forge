from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.command_lib.kuberun import pretty_print
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.core.resource import resource_printer
from six.moves.urllib_parse import urlparse
class ApplicationStatusPrinter(cp.CustomPrinterBase):
    """Prints the KubeRun Application Status custom human-readable format."""

    @staticmethod
    def Register(parser):
        """Register this custom printer with the given parser."""
        resource_printer.RegisterFormatter(_PRINTER_FORMAT, ApplicationStatusPrinter, hidden=True)
        parser.display_info.AddFormat(_PRINTER_FORMAT)

    def Transform(self, record):
        """Transform ApplicationStatus into the output structure of marker classes.

    Args:
      record: a dict object

    Returns:
      lines formatted for output
    """
        status = record['status']
        results = [cp.Section([cp.Labeled([('Environment', record['environment']), ('Ingress IP', status.ingress_ip)])])]
        if len(status.modules) == 1:
            results.append(cp.Section([cp.Labeled([('Components', _ComponentTable(status.modules[0].components))])], max_column_width=25))
        else:
            results.append(cp.Section([cp.Labeled([('Components', _ModulesTable(status.modules))])], max_column_width=25))
        results.append(cp.Section(['\n', _INGRESS_EXPLANATION_TEMPLATE.format(status.ingress_ip)]))
        return cp.Lines(results)