from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util import completers
class BuildsCompleter(completers.ListCommandCompleter):

    def __init__(self, **kwargs):
        super(BuildsCompleter, self).__init__(collection='cloudbuild.projects.builds', list_command='container builds list --uri', **kwargs)