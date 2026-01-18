from troveclient import base
from troveclient import common
@staticmethod
def _get_account_name(account):
    try:
        if account.name:
            return account.name
    except AttributeError:
        return account