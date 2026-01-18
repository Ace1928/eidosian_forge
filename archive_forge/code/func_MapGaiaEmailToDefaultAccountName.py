from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
def MapGaiaEmailToDefaultAccountName(email):
    """Returns the default account name given a GAIA email."""
    account_name = email.partition('@')[0].lower()
    if not account_name:
        raise GaiaException('Invalid email address [{email}].'.format(email=email))
    account_name = ''.join([char if char.isalnum() else '_' for char in account_name])
    if not account_name[0].isalpha():
        account_name = 'g' + account_name
    return account_name[:_MAX_ACCOUNT_NAME_LENGTH]