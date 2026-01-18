from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import subprocess
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.code import cross_platform_temp_file
from googlecloudsdk.command_lib.code import flags
from googlecloudsdk.command_lib.code import kubernetes
from googlecloudsdk.command_lib.code import local
from googlecloudsdk.command_lib.code import local_files
from googlecloudsdk.command_lib.code import run_subprocess
from googlecloudsdk.command_lib.code import skaffold
from googlecloudsdk.command_lib.code import yaml_helper
from googlecloudsdk.command_lib.code.cloud import artifact_registry
from googlecloudsdk.command_lib.code.cloud import cloud
from googlecloudsdk.command_lib.code.cloud import cloud_files
from googlecloudsdk.command_lib.code.cloud import cloudrun
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
import portpicker
import six
def _PrintDependencyVersions(args):
    """Print the version strings of the dependencies."""
    dependency_versions = {'skaffold': skaffold.GetVersion()}
    if args.IsSpecified('kube_context'):
        pass
    else:
        dependency_versions['minikube'] = kubernetes.GetMinikubeVersion()
    for name, version in sorted(dependency_versions.items()):
        print('%s: %s\n' % (name, version))