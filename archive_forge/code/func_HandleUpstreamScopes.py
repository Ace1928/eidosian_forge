from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.command_lib.container.fleet.features import base as feature_base
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.util import times
import six
def HandleUpstreamScopes(self, cluster_upgrade_spec):
    """Updates the Cluster Upgrade Feature's upstreamScopes field based on provided arguments.
    """
    if self.args.IsKnownAndSpecified('reset_upstream_scope') and self.args.reset_upstream_scope:
        cluster_upgrade_spec.upstreamScopes = []
    elif self.args.IsKnownAndSpecified('upstream_scope') and self.args.upstream_scope is not None:
        cluster_upgrade_spec.upstreamScopes = [self.args.upstream_scope]