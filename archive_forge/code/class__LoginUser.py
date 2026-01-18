import json
import time
from libcloud.common.types import LibcloudError
from libcloud.common.exceptions import BaseHTTPError
class _LoginUser:

    def __init__(self, user_id, auth=None):
        self.user_id = user_id
        self.auth = auth

    def to_dict(self):
        login_user = {'username': self.user_id}
        if self.auth is not None:
            login_user['ssh_keys'] = {'ssh_key': [self.auth.pubkey]}
        else:
            login_user['create_password'] = 'yes'
        return login_user