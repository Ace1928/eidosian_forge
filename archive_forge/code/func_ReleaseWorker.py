from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import dataclasses
import functools
import random
import string
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.run import condition as run_condition
from googlecloudsdk.api_lib.run import configuration
from googlecloudsdk.api_lib.run import domain_mapping
from googlecloudsdk.api_lib.run import execution
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run import job
from googlecloudsdk.api_lib.run import metric_names
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import route
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.run import task
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.run import artifact_registry
from googlecloudsdk.command_lib.run import config_changes as config_changes_mod
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import messages_util
from googlecloudsdk.command_lib.run import name_generator
from googlecloudsdk.command_lib.run import op_pollers
from googlecloudsdk.command_lib.run import resource_name_conversion
from googlecloudsdk.command_lib.run import stages
from googlecloudsdk.command_lib.run.sourcedeploys import deployer
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
import six
def ReleaseWorker(self, worker_ref, config_changes, release_track, tracker=None, asyn=False, for_replace=False, prefetch=False, build_image=None, build_pack=None, build_source=None, repo_to_create=None, already_activated_services=False, dry_run=False, generate_name=False):
    """Stubbed method for worker deploy surface.

    Args:
      worker_ref: Resource, the worker to release.
      config_changes: list, objects that implement Adjust().
      release_track: ReleaseTrack, the release track of a command calling this.
      tracker: StagedProgressTracker, to report on the progress of releasing.
      asyn: bool, if True, return without waiting for the service to be updated.
      for_replace: bool, If the change is for a replacing the service from a
        YAML specification.
      prefetch: the service, pre-fetched for ReleaseService. `False` indicates
        the caller did not perform a prefetch; `None` indicates a nonexistent
        service.
      build_image: The build image reference to the build.
      build_pack: The build pack reference to the build.
      build_source: The build source reference to the build.
      repo_to_create: Optional
        googlecloudsdk.command_lib.artifacts.docker_util.DockerRepo defining a
        repository to be created.
      already_activated_services: bool. If true, skip activation prompts for
        services
      dry_run: bool. If true, only validate the configuration.
      generate_name: bool. If true, create a revision name, otherwise add nonce.

    For private preview Worker is still Service underneath.

    Returns:
      service.Service, the service as returned by the server on the POST/PUT
       request to create/update the service.
    """
    return self.ReleaseService(worker_ref, config_changes, release_track, tracker=tracker, asyn=asyn, for_replace=for_replace, prefetch=prefetch, build_image=build_image, build_pack=build_pack, build_source=build_source, repo_to_create=repo_to_create, already_activated_services=already_activated_services, dry_run=dry_run, generate_name=generate_name)