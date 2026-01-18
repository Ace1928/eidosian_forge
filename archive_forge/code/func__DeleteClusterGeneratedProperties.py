from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
import textwrap
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import utils as api_utils
from googlecloudsdk.api_lib.dataproc import compute_helpers
from googlecloudsdk.api_lib.dataproc import constants
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
import six
from stdin.
def _DeleteClusterGeneratedProperties(cluster, dataproc):
    """Removes Dataproc generated properties from GCE-based Clusters."""
    props = encoding.MessageToPyValue(cluster.config.softwareConfig.properties)
    prop_prefixes_to_delete = ('hdfs:dfs.namenode.lifeline.rpc-address', 'hdfs:dfs.namenode.servicerpc-address')
    prop_keys_to_delete = [prop_key for prop_key in props.keys() if prop_key.startswith(prop_prefixes_to_delete)]
    for prop in prop_keys_to_delete:
        del props[prop]
    if not props:
        cluster.config.softwareConfig.properties = None
    else:
        cluster.config.softwareConfig.properties = encoding.DictToAdditionalPropertyMessage(props, dataproc.messages.SoftwareConfig.PropertiesValue)