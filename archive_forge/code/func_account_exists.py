from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
def account_exists(self, account):
    """
            Return account_id if account exists for given account id or name
            Raises an exception if account does not exist

            :param account: Account ID or Name
            :type account: str
            :return: Account ID if found, None if not found
        """
    if account.isdigit():
        account_id = int(account)
        try:
            result = self.elem_connect.get_account_by_id(account_id=account_id)
            if result.account.account_id == account_id:
                return account_id
        except solidfire.common.ApiServerError:
            pass
    result = self.elem_connect.get_account_by_name(username=account)
    return result.account.account_id