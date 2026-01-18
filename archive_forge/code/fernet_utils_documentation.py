import base64
import os
import stat
from cryptography import fernet
from oslo_log import log
from keystone.common import utils
import keystone.conf
Load keys from disk into a list.

        The first key in the list is the primary key used for encryption. All
        other keys are active secondary keys that can be used for decrypting
        tokens.

        :param use_null_key: If true, a known key containing null bytes will be
                             appended to the list of returned keys.

        