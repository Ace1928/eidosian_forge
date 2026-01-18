import enum
import os.path
from googlecloudsdk.api_lib.run import api_enabler
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.run import artifact_registry
from googlecloudsdk.command_lib.run import config_changes
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import container_parser
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import messages_util
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.command_lib.run import resource_args
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.run import stages
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
def _BuildFromSource(self, args, build_from_source, worker_ref, already_activated_services):
    container = next(iter(build_from_source.values()))
    pack = None
    repo_to_create = None
    source = container.source
    ar_repo = docker_util.DockerRepo(project_id=properties.VALUES.core.project.Get(required=True), location_id=artifact_registry.RepoRegion(args), repo_id='cloud-run-source-deploy')
    if artifact_registry.ShouldCreateRepository(ar_repo, skip_activation_prompt=already_activated_services):
        repo_to_create = ar_repo
    container.image = '{repo}/{worker}'.format(repo=ar_repo.GetDockerString(), worker=worker_ref.servicesId)
    docker_file = source + '/Dockerfile'
    if os.path.exists(docker_file):
        build_type = BuildType.DOCKERFILE
    else:
        pack = _CreateBuildPack(container)
        build_type = BuildType.BUILDPACKS
    image = None if pack else container.image
    if flags.FlagIsExplicitlySet(args, 'delegate_builds'):
        image = pack[0].get('image') if pack else image
    operation_message = 'Building using {build_type} and deploying container to'.format(build_type=build_type.value)
    pretty_print.Info(messages_util.GetBuildEquivalentForSourceRunMessage(worker_ref.servicesId, pack, source, subgroup='workers '))
    return (image, pack, source, operation_message, repo_to_create)