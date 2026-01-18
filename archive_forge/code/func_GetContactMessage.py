from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
def GetContactMessage(version=DEFAULT_API_VERSION):
    """Gets the contact message for the specified version of the API."""
    versioned_message_type = _CONTACT_TYPES_BY_VERSION[version]['message_name']
    return getattr(GetMessages(version), versioned_message_type)