from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.accesscontextmanager import zones as zones_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.accesscontextmanager import perimeters
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.args import repeated
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class CreatePerimeterDryRun(base.UpdateCommand):
    """Creates a dry-run spec for a new or existing Service Perimeter."""
    _API_VERSION = 'v1'

    @staticmethod
    def Args(parser):
        CreatePerimeterDryRun.ArgsVersioned(parser, version='v1')

    @staticmethod
    def ArgsVersioned(parser, version='v1'):
        parser.add_argument('--async', action='store_true', help='Return immediately, without waiting for the operation in\n                progress to complete.')
        perimeters.AddResourceArg(parser, 'to update')
        top_level_group = parser.add_mutually_exclusive_group(required=True)
        existing_perimeter_group = top_level_group.add_argument_group('Arguments for creating dry-run spec for an **existing** Service Perimeter.')
        _AddCommonArgsForDryRunCreate(existing_perimeter_group, version=version)
        new_perimeter_group = top_level_group.add_argument_group('Arguments for creating a dry-run spec for a new Service Perimeter.')
        _AddCommonArgsForDryRunCreate(new_perimeter_group, prefix='perimeter-', version=version)
        new_perimeter_group.add_argument('--perimeter-title', required=True, default=None, help='Short human-readable title for the Service Perimeter.')
        new_perimeter_group.add_argument('--perimeter-description', default=None, help='Long-form description of Service Perimeter.')
        new_perimeter_group.add_argument('--perimeter-type', required=True, default=None, help='Type of the perimeter.\n\n            A *regular* perimeter allows resources within this service perimeter\n            to import and export data amongst themselves. A project may belong\n            to at most one regular service perimeter.\n\n            A *bridge* perimeter allows resources in different regular service\n            perimeters to import and export data between each other. A project\n            may belong to multiple bridge service perimeters (only if it also\n            belongs to a regular service perimeter). Both restricted and\n            unrestricted service lists, as well as access level lists, must be\n            empty.')

    def Run(self, args):
        client = zones_api.Client(version=self._API_VERSION)
        perimeter_ref = args.CONCEPTS.perimeter.Parse()
        perimeter_type = perimeters.GetPerimeterTypeEnumForShortName(args.perimeter_type, self._API_VERSION)
        resources = _ParseArgWithShortName(args, 'resources')
        levels = _ParseArgWithShortName(args, 'access_levels')
        levels = perimeters.ExpandLevelNamesIfNecessary(levels, perimeter_ref.accessPoliciesId)
        restricted_services = _ParseArgWithShortName(args, 'restricted_services')
        vpc_allowed_services = _ParseArgWithShortName(args, 'vpc_allowed_services')
        ingress_policies, egress_policies = _ParseDirectionalPolicies(args)
        if args.enable_vpc_accessible_services is None and args.perimeter_enable_vpc_accessible_services is None:
            enable_vpc_accessible_services = None
        else:
            enable_vpc_accessible_services = args.enable_vpc_accessible_services or args.perimeter_enable_vpc_accessible_services
        result = repeated.CachedResult.FromFunc(client.Get, perimeter_ref)
        try:
            result.Get()
        except apitools_exceptions.HttpNotFoundError:
            if args.perimeter_title is None or perimeter_type is None:
                raise exceptions.RequiredArgumentException('perimeter-title', 'Since this Service Perimeter does not exist, perimeter-title and perimeter-type must be supplied.')
        else:
            if args.perimeter_title is not None or perimeter_type is not None:
                raise exceptions.InvalidArgumentException('perimeter-title', 'A Service Perimeter with the given name already exists. The title and the type fields cannot be updated in the dry-run mode.')
        policies.ValidateAccessPolicyArg(perimeter_ref, args)
        return client.PatchDryRunConfig(perimeter_ref, title=args.perimeter_title, description=args.perimeter_description, perimeter_type=perimeter_type, resources=resources, levels=levels, restricted_services=restricted_services, vpc_allowed_services=vpc_allowed_services, enable_vpc_accessible_services=enable_vpc_accessible_services, ingress_policies=ingress_policies, egress_policies=egress_policies)