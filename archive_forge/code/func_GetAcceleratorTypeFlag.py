from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def GetAcceleratorTypeFlag():
    """Set argument for choosing the TPU Accelerator type."""
    return base.Argument('--accelerator-type', default='v2-8', type=lambda x: x.lower(), required=False, help='      TPU accelerator type for the TPU.\n       If not specified, this defaults to `v2-8`.\n      ')