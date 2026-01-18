from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import client
from googlecloudsdk.command_lib.util.args import labels_util
def HandleNamespaceLabelsUpdateRequest(ref, args):
    """Add namespace labels to update request.

  Args:
    ref: reference to the scope object.
    args: command line arguments.

  Returns:
    response

  """
    mask = []
    release_track = args.calliope_command.ReleaseTrack()
    fleetclient = client.FleetClient(release_track)
    labels_diff = labels_util.Diff.FromUpdateArgs(args)
    namespace_labels_diff = labels_util.Diff(args.update_namespace_labels, args.remove_namespace_labels, args.clear_namespace_labels)
    current_scope = fleetclient.GetScope(ref.RelativeName())
    new_labels = labels_diff.Apply(fleetclient.messages.Scope.LabelsValue, current_scope.labels).GetOrNone()
    if new_labels:
        mask.append('labels')
    new_namespace_labels = namespace_labels_diff.Apply(fleetclient.messages.Scope.NamespaceLabelsValue, current_scope.namespaceLabels).GetOrNone()
    if new_namespace_labels:
        mask.append('namespace_labels')
    if not mask:
        response = fleetclient.messages.Scope(name=ref.RelativeName())
        return response
    return fleetclient.UpdateScope(ref.RelativeName(), new_labels, new_namespace_labels, ','.join(mask))