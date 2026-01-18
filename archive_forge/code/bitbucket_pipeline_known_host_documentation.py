from __future__ import absolute_import, division, print_function
import socket
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.source_control.bitbucket import BitbucketHelper

    Fetches public key for specified host

    :param module: instance of the :class:`AnsibleModule`
    :param hostname: host name
    :return: key type and key content
    :rtype: tuple

    Return example::

        (
            'ssh-rsa',
            'AAAAB3NzaC1yc2EAAAABIwAAA...SBne8+seeFVBoGqzHM9yXw==',
        )
    