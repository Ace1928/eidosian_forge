from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files as file_utils
def WarnAlias(alias):
    """WarnAlias outputs a warning telling users to not use the given alias."""
    msg = 'Image aliases are deprecated and will be removed in a future version. '
    if alias.family is not None:
        msg += 'Please use --image-family={family} and --image-project={project} instead.'.format(family=alias.family, project=alias.project)
    else:
        msg += 'Please use --image-family and --image-project instead.'
    log.warning(msg)