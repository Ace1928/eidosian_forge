import datetime
import urllib
import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.extras._saml2.v3 import base
def _token_dates(self, fmt='%Y-%m-%dT%H:%M:%S.%fZ'):
    """Calculate created and expires datetime objects.

        The method is going to be used for building ADFS Request Security
        Token message. Time interval between ``created`` and ``expires``
        dates is now static and equals to 120 seconds. ADFS security tokens
        should not be live too long, as currently ``keystoneauth1``
        doesn't have mechanisms for reusing such tokens (every time ADFS authn
        method is called, keystoneauth1 will login with the ADFS instance).

        :param fmt: Datetime format for specifying string format of a date.
                    It should not be changed if the method is going to be used
                    for building the ADFS security token request.
        :type fmt: string

        """
    date_created = datetime.datetime.utcnow()
    date_expires = date_created + datetime.timedelta(seconds=self.DEFAULT_ADFS_TOKEN_EXPIRATION)
    return [_time.strftime(fmt) for _time in (date_created, date_expires)]