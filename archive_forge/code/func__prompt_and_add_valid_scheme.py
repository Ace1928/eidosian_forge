from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.transfer import creds_util
from googlecloudsdk.command_lib.transfer import jobs_flag_util
from googlecloudsdk.command_lib.transfer import name_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
def _prompt_and_add_valid_scheme(url, valid_schemes):
    """Has user select a valid scheme from a list and returns new URL."""
    if not console_io.CanPrompt():
        raise errors.InvalidUrlError('Did you mean "posix://{}"'.format(url.object_name))
    scheme_index = console_io.PromptChoice([scheme.value + '://' for scheme in valid_schemes], cancel_option=True, message='Storage Transfer does not support direct file URLs: {}\nDid you mean to use "posix://"?\nRun this command with "--help" for more info,\nor select a valid scheme below.'.format(url))
    new_scheme = valid_schemes[scheme_index]
    return storage_url.switch_scheme(url, new_scheme)