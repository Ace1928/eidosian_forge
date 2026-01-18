from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.run import run_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.builds import submit_util
from googlecloudsdk.command_lib.run import stages
from googlecloudsdk.command_lib.run.sourcedeploys import sources
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _PrepareBuildConfig(tracker, build_image, build_source, build_pack, release_track, region, resource_ref):
    """Upload the provided build source and prepare build config for cloud build."""
    tracker.StartStage(stages.UPLOAD_SOURCE)
    tracker.UpdateHeaderMessage('Uploading sources.')
    build_messages = cloudbuild_util.GetMessagesModule()
    if release_track is base.ReleaseTrack.ALPHA:
        source = sources.Upload(build_source, region, resource_ref)
        uri = f'gs://{source.bucket}/{source.name}#{source.generation}'
        if build_pack is not None:
            envs = build_pack[0].get('envs', [])
            envs.append(f'GOOGLE_LABEL_SOURCE={uri}')
            build_pack[0].update({'envs': envs})
        properties.VALUES.builds.use_kaniko.Set(False)
        build_config = submit_util.CreateBuildConfig(build_image, no_cache=False, messages=build_messages, substitutions=None, arg_config=None, is_specified_source=True, no_source=False, source=build_source, gcs_source_staging_dir=None, ignore_file=None, arg_gcs_log_dir=None, arg_machine_type=None, arg_disk_size=None, arg_worker_pool=None, arg_dir=None, arg_revision=None, arg_git_source_dir=None, arg_git_source_revision=None, arg_service_account=None, buildpack=build_pack, hide_logs=True, skip_set_source=True, client_tag='gcloudrun')
        if build_pack is None:
            assert build_config.steps[0].name == 'gcr.io/cloud-builders/docker'
            build_config.steps[0].args.extend(['--label', f'google.source={uri}'])
        build_config.source = build_messages.Source(storageSource=build_messages.StorageSource(bucket=source.bucket, object=source.name, generation=source.generation))
    else:
        properties.VALUES.builds.use_kaniko.Set(False)
        build_config = submit_util.CreateBuildConfig(build_image, no_cache=False, messages=build_messages, substitutions=None, arg_config=None, is_specified_source=True, no_source=False, source=build_source, gcs_source_staging_dir=None, ignore_file=None, arg_gcs_log_dir=None, arg_machine_type=None, arg_disk_size=None, arg_worker_pool=None, arg_dir=None, arg_revision=None, arg_git_source_dir=None, arg_git_source_revision=None, arg_service_account=None, buildpack=build_pack, hide_logs=True, client_tag='gcloudrun')
    tracker.CompleteStage(stages.UPLOAD_SOURCE)
    return (build_messages, build_config)