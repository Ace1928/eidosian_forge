from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import compute_helpers
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataproc import clusters
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.util.args import labels_util
class SetManagedCluster(base.UpdateCommand):
    """Set a managed cluster for the workflow template."""
    detailed_help = {'EXAMPLES': '\nTo update managed cluster in a workflow template, run:\n\n  $ {command} my_template --region=us-central1 --no-address --num-workers=10 --worker-machine-type=custom-6-23040\n\n'}

    @classmethod
    def Args(cls, parser):
        dataproc = dp.Dataproc(cls.ReleaseTrack())
        parser.add_argument('--cluster-name', help='          The name of the managed dataproc cluster.\n          If unspecified, the workflow template ID will be used.')
        clusters.ArgsForClusterRef(parser, dataproc, cls.Beta(), cls.Alpha(), include_deprecated=cls.Beta(), include_gke_platform_args=False)
        flags.AddTemplateResourceArg(parser, 'set managed cluster', dataproc.api_version)
        if cls.Beta():
            clusters.BetaArgsForClusterRef(parser)

    @classmethod
    def Beta(cls):
        return cls.ReleaseTrack() != base.ReleaseTrack.GA

    @classmethod
    def Alpha(cls):
        return cls.ReleaseTrack() == base.ReleaseTrack.ALPHA

    @classmethod
    def GetComputeReleaseTrack(cls):
        if cls.Beta():
            return base.ReleaseTrack.BETA
        return base.ReleaseTrack.GA

    def Run(self, args):
        dataproc = dp.Dataproc(self.ReleaseTrack())
        template_ref = args.CONCEPTS.template.Parse()
        workflow_template = dataproc.GetRegionsWorkflowTemplate(template_ref, args.version)
        if args.cluster_name:
            cluster_name = args.cluster_name
        else:
            cluster_name = template_ref.workflowTemplatesId
        compute_resources = compute_helpers.GetComputeResources(self.GetComputeReleaseTrack(), cluster_name, template_ref.regionsId)
        cluster_config = clusters.GetClusterConfig(args, dataproc, template_ref.projectsId, compute_resources, self.Beta(), self.Alpha(), include_deprecated=self.Beta())
        labels = labels_util.ParseCreateArgs(args, dataproc.messages.ManagedCluster.LabelsValue)
        managed_cluster = dataproc.messages.ManagedCluster(clusterName=cluster_name, config=cluster_config, labels=labels)
        workflow_template.placement = dataproc.messages.WorkflowTemplatePlacement(managedCluster=managed_cluster)
        response = dataproc.client.projects_regions_workflowTemplates.Update(workflow_template)
        return response