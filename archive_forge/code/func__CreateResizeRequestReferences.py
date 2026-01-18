from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
def _CreateResizeRequestReferences(self, resize_requests, igm_ref, resources):
    resize_request_references = []
    if igm_ref.Collection() == 'compute.instanceGroupManagers':
        for resize_request_name in resize_requests:
            resize_request_references.append(resources.Parse(resize_request_name, {'project': igm_ref.project, 'zone': igm_ref.zone, 'instanceGroupManager': igm_ref.instanceGroupManager}, collection='compute.instanceGroupManagerResizeRequests'))
        return resize_request_references
    raise ValueError('Unknown reference type {0}'.format(igm_ref.Collection()))