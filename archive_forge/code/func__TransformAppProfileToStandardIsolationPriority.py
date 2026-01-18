from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.bigtable import app_profiles
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bigtable import arguments
def _TransformAppProfileToStandardIsolationPriority(app_profile):
    """Extracts the Data Boot compute billing owner from the app profile."""
    if 'dataBoostIsolationReadOnly' in app_profile:
        return ''
    elif 'standardIsolation' in app_profile and 'priority' in app_profile['standardIsolation']:
        return app_profile['standardIsolation']['priority']
    else:
        return 'PRIORITY_HIGH'