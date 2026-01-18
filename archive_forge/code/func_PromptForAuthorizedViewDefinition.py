from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import copy
import io
import json
import textwrap
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_diff
from googlecloudsdk.core.util import edit
import six
def PromptForAuthorizedViewDefinition(is_create, pre_encoded, current_authorized_view=None):
    """Prompt user to fill out a JSON/YAML format representation of an authorized view.

  Returns the parsed authorized view proto message from user's response.

  Args:
    is_create: True if the prompt is for creating an authorized view. False if
      the prompt is for updating an authorized view.
    pre_encoded: True if all binary fields in the authorized view definition are
      already Base64-encoded. We skip the step of applying Base64 encoding in
      this case.
    current_authorized_view: The current authorized view definition. Only used
      in the update case to be included as part of the initial commented prompt.

  Returns:
    an authorized view proto message with fields filled accordingly.

  Raises:
    ChildProcessError if the user did not save the temporary file.
    ChildProcessError if there is a problem running the editor.
    ValueError if the user's response does not follow YAML or JSON format.
    ValueError if the YAML/JSON object cannot be parsed as a valid authorized
      View.
  """
    authorized_view_message_type = util.GetAdminMessages().AuthorizedView
    if is_create:
        help_text = BuildCreateAuthorizedViewFileContents()
    else:
        help_text = BuildUpdateAuthorizedViewFileContents(current_authorized_view, pre_encoded)
    try:
        content = edit.OnlineEdit(help_text)
    except edit.NoSaveException:
        raise ChildProcessError('Edit aborted by user.')
    except edit.EditorException as e:
        raise ChildProcessError('There was a problem applying your changes. [{0}].'.format(e))
    try:
        authorized_view_to_parse = yaml.load(content)
        if not pre_encoded:
            Base64EncodingYamlAuthorizedViewDefinition(authorized_view_to_parse)
        authorized_view = encoding.PyValueToMessage(authorized_view_message_type, authorized_view_to_parse)
    except yaml.YAMLParseError as e:
        raise ValueError('Provided response is not a properly formatted YAML or JSON file. [{0}].'.format(e))
    except AttributeError as e:
        raise ValueError('Provided response cannot be parsed as a valid authorized view. [{0}].'.format(e))
    return authorized_view