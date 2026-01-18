from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.util.args import labels_util
class SetClusterSelector(base.UpdateCommand):
    """Set cluster selector for the workflow template."""
    detailed_help = {'EXAMPLES': '\nTo set placement cluster selector labels on a workflow template, run:\n\n  $ {command} my_template --region=us-central1 --cluster-labels=environment=production\n'}

    @classmethod
    def Args(cls, parser):
        dataproc = dp.Dataproc(cls.ReleaseTrack())
        parser.add_argument('--cluster-labels', metavar='KEY=VALUE', type=arg_parsers.ArgDict(key_type=labels_util.KEY_FORMAT_VALIDATOR, value_type=labels_util.VALUE_FORMAT_VALIDATOR, min_length=1), action=arg_parsers.UpdateAction, help='A list of label KEY=VALUE pairs to add.')
        flags.AddTemplateResourceArg(parser, 'set cluster selector', dataproc.api_version)

    def Run(self, args):
        dataproc = dp.Dataproc(self.ReleaseTrack())
        template_ref = args.CONCEPTS.template.Parse()
        workflow_template = dataproc.GetRegionsWorkflowTemplate(template_ref, args.version)
        labels = labels_util.Diff(additions=args.cluster_labels).Apply(dataproc.messages.ClusterSelector.ClusterLabelsValue).GetOrNone()
        cluster_selector = dataproc.messages.ClusterSelector(clusterLabels=labels)
        workflow_template.placement = dataproc.messages.WorkflowTemplatePlacement(clusterSelector=cluster_selector)
        response = dataproc.client.projects_regions_workflowTemplates.Update(workflow_template)
        return response