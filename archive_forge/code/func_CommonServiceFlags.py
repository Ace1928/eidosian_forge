from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def CommonServiceFlags(is_create=False):
    return FlagGroup(NamespaceFlag(), ImageFlag(required=is_create), CPUFlag(), GPUFlag(), MemoryFlag(), PortFlag(), Http2Flag(), ConcurrencyFlag(), EntrypointFlags(), ScalingFlags(), LabelsFlags(set_flag_only=is_create), ConfigMapFlags(set_flag_only=is_create), SecretsFlags(set_flag_only=is_create), EnvVarsFlags(set_flag_only=is_create), ConnectivityFlag(), ServiceAccountFlag(), RevisionSuffixFlag(), TimeoutFlag())