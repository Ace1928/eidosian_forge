from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def GetCustomFeaturesFlag():
    """Returns the flag for specifying custom features in an SSL policy."""
    return base.Argument('--custom-features', metavar='CUSTOM_FEATURES', type=arg_parsers.ArgList(), help='A comma-separated list of custom features, required when the profile being used is CUSTOM.\n\nUsing CUSTOM profile allows customization of the features that are part of the SSL policy. This flag allows specifying those custom features.\n\nThe list of all supported custom features can be obtained using:\n\n  gcloud compute ssl-policies list-available-features\n')