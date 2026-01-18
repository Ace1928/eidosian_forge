from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import filter_scope_rewriter
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_expr_rewrite
from googlecloudsdk.core.resource import resource_projector
import six
class AllScopes(object):
    """Holds information about wildcard use of list command."""

    def __init__(self, projects, zonal, regional):
        self.projects = projects
        self.zonal = zonal
        self.regional = regional

    def __eq__(self, other):
        if not isinstance(other, AllScopes):
            return False
        return self.projects == other.projects and self.zonal == other.zonal and (self.regional == other.regional)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.projects) ^ hash(self.zonal) ^ hash(self.regional)

    def __repr__(self):
        return 'AllScopes(projects={}, zonal={}, regional={})'.format(repr(self.projects), repr(self.zonal), repr(self.regional))