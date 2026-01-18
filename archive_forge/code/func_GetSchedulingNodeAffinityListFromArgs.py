from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
def GetSchedulingNodeAffinityListFromArgs(args, messages, support_node_project=False):
    """Returns a list of ScheduleNodeAffinity messages populated from args."""
    operator_enum = messages.SchedulingNodeAffinity.OperatorValueValuesEnum
    node_affinities = []
    if args.IsSpecified('node_affinity_file'):
        affinities_yaml = yaml.load(args.node_affinity_file)
        if not affinities_yaml:
            raise NodeAffinityFileParseError('No node affinity labels specified. You must specify at least one label to create a sole tenancy instance.')
        for affinity in affinities_yaml:
            if not affinity:
                raise NodeAffinityFileParseError('Empty list item in JSON/YAML file.')
            try:
                node_affinity = encoding.PyValueToMessage(messages.SchedulingNodeAffinity, affinity)
            except Exception as e:
                raise NodeAffinityFileParseError(e)
            if not node_affinity.key:
                raise NodeAffinityFileParseError('A key must be specified for every node affinity label.')
            if node_affinity.all_unrecognized_fields():
                raise NodeAffinityFileParseError('Key [{0}] has invalid field formats for: {1}'.format(node_affinity.key, node_affinity.all_unrecognized_fields()))
            node_affinities.append(node_affinity)
    if args.IsSpecified('node_group'):
        node_affinities.append(messages.SchedulingNodeAffinity(key='compute.googleapis.com/node-group-name', operator=operator_enum.IN, values=[args.node_group]))
    if args.IsSpecified('node'):
        node_affinities.append(messages.SchedulingNodeAffinity(key='compute.googleapis.com/node-name', operator=operator_enum.IN, values=[args.node]))
    if support_node_project and args.IsSpecified('node_project'):
        node_affinities.append(messages.SchedulingNodeAffinity(key='compute.googleapis.com/project', operator=operator_enum.IN, values=[args.node_project]))
    return node_affinities