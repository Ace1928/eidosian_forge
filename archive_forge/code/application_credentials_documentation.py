from keystoneclient import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
from keystoneclient import utils
Delete an application credential.

        :param application_credential: the application credential to be deleted
        :type credential: str or
            :class:`keystoneclient.v3.application_credentials.ApplicationCredential`

        :returns: response object with 204 status
        :rtype: :class:`requests.models.Response`

        