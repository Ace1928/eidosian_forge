import keystone.conf
from keystone.common import fernet_utils as utils
from keystone.credential.providers import fernet as credential_fernet
Credential key repository is empty.

    After configuring keystone to use the Fernet credential provider, you
    should use `keystone-manage credential_setup` to initially populate your
    key repository with keys, and periodically rotate your keys with
    `keystone-manage credential_rotate`.
    