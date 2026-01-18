from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import random
import string
import time
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py.exceptions import HttpError
from apitools.base.py.exceptions import HttpNotFoundError
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import logs as cb_logs
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.services import enable_api as services_api
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as http_exc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.cloudbuild import execution
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding as encoding_util
import six
def AddAWSImageImportSourceArgs(aws_group):
    """Adds args for image import from AWS."""
    aws_group.add_argument('--aws-access-key-id', required=True, help='          Access key ID for a temporary AWS credential.\n          This ID must be generated using the AWS Security Token Service.\n          ')
    aws_group.add_argument('--aws-secret-access-key', required=True, help='          Secret access key for a temporary AWS credential.\n          This key must be generated using the AWS Security Token Service.\n          ')
    aws_group.add_argument('--aws-session-token', required=True, help='          Session token for a temporary AWS credential. This session\n          token must be generated using the AWS Security Token Service.\n          ')
    aws_group.add_argument('--aws-region', required=True, help='AWS region of the image that you want to import.')
    step_to_begin = aws_group.add_mutually_exclusive_group(required=True, help='          Specify whether to import from an AMI or disk image.\n          ')
    begin_from_export = step_to_begin.add_group(help='      If importing an AMI,  specify the following two flags:')
    begin_from_export.add_argument('--aws-ami-id', required=True, help='AWS AMI ID of the image to import.')
    begin_from_export.add_argument('--aws-ami-export-location', required=True, help='          An AWS S3 bucket location where the converted image file can be\n          temporarily exported to before the import to Cloud Storage.')
    begin_from_file = step_to_begin.add_group(help='      If importing a disk image,  specify the following:')
    begin_from_file.add_argument('--aws-source-ami-file-path', help='          S3 resource path of the exported image file that you want\n          to import.\n          ')