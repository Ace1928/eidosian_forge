import enum
import os
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import kubeconfig
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.run import traffic
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions as services_exceptions
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.functions.v2.deploy import env_vars_util
from googlecloudsdk.command_lib.run import config_changes
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.command_lib.run import resource_args
from googlecloudsdk.command_lib.run import secrets_mapping
from googlecloudsdk.command_lib.run import volumes
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
def SourceArg():
    return base.Argument('--source', help='The location of the source to build. If a Dockerfile is present in the source code directory, it will be built using that Dockerfile, otherwise it will use Google Cloud buildpacks. See https://cloud.google.com/run/docs/deploying-source-code for more details. The location can be a directory on a local disk or a gzipped archive file (.tar.gz) in Google Cloud Storage. If the source is a local directory, this command skips the files specified in the `--ignore-file`. If `--ignore-file` is not specified, use `.gcloudignore` file. If a `.gcloudignore` file is absent and a `.gitignore` file is present in the local source directory, gcloud will use a generated Git-compatible `.gcloudignore` file that respects your .gitignored files. The global `.gitignore` is not respected. For more information on `.gcloudignore`, see `gcloud topic gcloudignore`.')