from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import json
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
def _get_format(self, credential_source_type, credential_source_field_name):
    """Returns an optional dictionary indicating the format of the token.

    This is a shared method, that several different token types need access to.

    Args:
      credential_source_type: The format of the token, either 'json' or 'text'.
      credential_source_field_name: The field name of a JSON object containing
        the text version of the token.

    Raises:
      GeneratorError: if an invalid token format is specified, or no field name
      is specified for a json token.

    """
    if not credential_source_type:
        return None
    credential_source_type = credential_source_type.lower()
    if credential_source_type not in ('json', 'text'):
        raise GeneratorError('--credential-source-type must be either "json" or "text"')
    token_format = {'type': credential_source_type}
    if credential_source_type == 'json':
        if not credential_source_field_name:
            raise GeneratorError('--credential-source-field-name required for JSON formatted tokens')
        token_format['subject_token_field_name'] = credential_source_field_name
    return token_format