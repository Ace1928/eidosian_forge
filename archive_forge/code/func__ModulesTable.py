from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.command_lib.kuberun import pretty_print
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.core.resource import resource_printer
from six.moves.urllib_parse import urlparse
def _ModulesTable(modules):
    """Format module to table."""
    con = console_attr.GetConsoleAttr()
    rows = []
    for module in modules:
        for component in module.components:
            status_symbol = pretty_print.GetReadySymbol(component.deployment_state)
            status_color = pretty_print.GetReadyColor(component.deployment_state)
            rows.append((con.Colorize(status_symbol, status_color), component.name, module.name, component.deployment_reason, component.commit_id[:6], component.deployment_time, component.url))
    return cp.Table([('', 'NAME', 'MODULE', 'REASON', 'COMMIT', 'LAST-DEPLOYED', 'URL')] + rows, console_attr=con)