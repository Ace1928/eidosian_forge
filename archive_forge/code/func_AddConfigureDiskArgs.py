from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.instance_templates import service_proxy_aux_data
def AddConfigureDiskArgs(parser):
    parser.add_argument('--configure-disk', type=arg_parsers.ArgDict(spec={'auto-delete': arg_parsers.ArgBoolean(), 'device-name': str, 'instantiate-from': str, 'custom-image': str}), metavar='PROPERTY=VALUE', action='append', help="    This option has effect only when used with `--source-instance`. It\n    allows you to override how the source-instance's disks are defined in\n    the template.\n\n    *device-name*::: Name of the device for which the configuration is being\n    overridden.\n\n    *auto-delete*::: If `true`, this persistent disk will be automatically\n    deleted when the instance is deleted. However, if the disk is\n    detached from the instance, this option won't apply. If not provided,\n    the setting is copied from the source instance. Allowed values of the\n    flag are: `false`, `no`, `true`, and `yes`.\n\n    *instantiate-from*::: Specifies whether to include the disk and which\n    image to use. Valid values are: {}\n\n    *custom-image*::: The custom image to use if custom-image is specified\n    for instantiate-from.\n    ".format(', '.join(_INSTANTIATE_FROM_VALUES)))