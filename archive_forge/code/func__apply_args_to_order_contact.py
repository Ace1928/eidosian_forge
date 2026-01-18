from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.transfer.appliances import flags
def _apply_args_to_order_contact(contact_field):
    """Maps command arguments to order contact values."""
    emails = contact_field.get('emails', [])
    return {'email': emails.pop(0), 'additionalEmails': emails, 'business': contact_field.get('business', None), 'contactName': contact_field.get('name', None), 'phone': contact_field.get('phone', None)}