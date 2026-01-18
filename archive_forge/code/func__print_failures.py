import collections
from osc_lib.command import command
from heatclient._i18n import _
from heatclient.common import format_utils
from heatclient import exc
def _print_failures(self, failures, deployment_failures, long=False):
    """Print failed resources.

        If the resource is a deployment resource, look up the deployment and
        print deploy_stdout and deploy_stderr.
        """
    out = self.app.stdout
    if not failures:
        return
    for k, f in failures.items():
        out.write('%s:\n' % k)
        out.write('  resource_type: %s\n' % f.resource_type)
        out.write('  physical_resource_id: %s\n' % f.physical_resource_id)
        out.write('  status: %s\n' % f.resource_status)
        reason = format_utils.indent_and_truncate(f.resource_status_reason, spaces=4, truncate=not long, truncate_prefix='...\n')
        out.write('  status_reason: |\n%s\n' % reason)
        df = deployment_failures.get(f.physical_resource_id)
        if df:
            for output in ('deploy_stdout', 'deploy_stderr'):
                format_utils.print_software_deployment_output(data=df.output_values, name=output, long=long, out=out)