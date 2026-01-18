from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.crypto.plugins.module_utils.acme.acme import (
from ansible_collections.community.crypto.plugins.module_utils.acme.account import (
from ansible_collections.community.crypto.plugins.module_utils.acme.challenges import (
from ansible_collections.community.crypto.plugins.module_utils.acme.certificates import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.io import (
from ansible_collections.community.crypto.plugins.module_utils.acme.orders import (
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import (

        Deactivates all valid authz's. Does not raise exceptions.
        https://community.letsencrypt.org/t/authorization-deactivation/19860/2
        https://tools.ietf.org/html/rfc8555#section-7.5.2
        