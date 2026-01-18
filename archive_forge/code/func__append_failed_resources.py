import collections
from osc_lib.command import command
from heatclient._i18n import _
from heatclient.common import format_utils
from heatclient import exc
def _append_failed_resources(self, failures, resources, resource_path):
    """Recursively build list of failed resources."""
    appended = False
    for rsc in resources:
        if not rsc.resource_status.endswith('FAILED'):
            continue
        links_rel = list([link['rel'] for link in rsc.links])
        is_nested = 'nested' in links_rel
        nested_appended = False
        next_resource_path = list(resource_path)
        next_resource_path.append(rsc.resource_name)
        if is_nested:
            try:
                nested_resources = self.heat_client.resources.list(rsc.physical_resource_id)
                nested_appended = self._append_failed_resources(failures, nested_resources, next_resource_path)
            except exc.HTTPNotFound:
                pass
        if not nested_appended:
            failures['.'.join(next_resource_path)] = rsc
        appended = True
    return appended