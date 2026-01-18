import base64
import binascii
import os
import re
from boto.compat import StringIO, six
from boto.exception import BotoClientError
from boto.s3.key import Key as S3Key
from boto.s3.keyfile import KeyFile
from boto.utils import compute_hash, get_utf8able_str
def add_group_email_grant(self, permission, email_address, headers=None):
    """
        Convenience method that provides a quick way to add an email group
        grant to a key. This method retrieves the current ACL, creates a new
        grant based on the parameters passed in, adds that grant to the ACL and
        then PUT's the new ACL back to GS.

        :type permission: string
        :param permission: The permission being granted. Should be one of:
            READ|FULL_CONTROL
            See http://code.google.com/apis/storage/docs/developer-guide.html#authorization
            for more details on permissions.

        :type email_address: string
        :param email_address: The email address associated with the Google
            Group to which you are granting the permission.
        """
    acl = self.get_acl(headers=headers)
    acl.add_group_email_grant(permission, email_address)
    self.set_acl(acl, headers=headers)