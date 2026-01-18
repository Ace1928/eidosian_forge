import os
import platform
import socket
import stat
import six
from oauthlib import oauth1
from six.moves.urllib.parse import parse_qs, urlencode
from lazr.restfulclient.authorize import HttpAuthorizer
from lazr.restfulclient.errors import CredentialsFileError
class OAuthAuthorizer(HttpAuthorizer):
    """A client that signs every outgoing request with OAuth credentials."""

    def __init__(self, consumer_name=None, consumer_secret='', access_token=None, oauth_realm='OAuth', application_name=None):
        self.consumer = None
        if consumer_name is not None:
            self.consumer = Consumer(consumer_name, consumer_secret, application_name)
        self.access_token = access_token
        self.oauth_realm = oauth_realm

    @property
    def user_agent_params(self):
        """Any information necessary to identify this user agent.

        In this case, the OAuth consumer name.
        """
        params = {}
        if self.consumer is None:
            return params
        params['oauth_consumer'] = self.consumer.key
        if self.consumer.application_name is not None:
            params['application'] = self.consumer.application_name
        return params

    def load(self, readable_file):
        """Load credentials from a file-like object.

        This overrides the consumer and access token given in the constructor
        and replaces them with the values read from the file.

        :param readable_file: A file-like object to read the credentials from
        :type readable_file: Any object supporting the file-like `read()`
            method
        """
        parser = SafeConfigParser()
        if hasattr(parser, 'read_file'):
            reader = parser.read_file
        else:
            reader = parser.readfp
        reader(readable_file)
        if not parser.has_section(CREDENTIALS_FILE_VERSION):
            raise CredentialsFileError('No configuration for version %s' % CREDENTIALS_FILE_VERSION)
        consumer_key = parser.get(CREDENTIALS_FILE_VERSION, 'consumer_key')
        consumer_secret = parser.get(CREDENTIALS_FILE_VERSION, 'consumer_secret')
        self.consumer = Consumer(consumer_key, consumer_secret)
        access_token = parser.get(CREDENTIALS_FILE_VERSION, 'access_token')
        access_secret = parser.get(CREDENTIALS_FILE_VERSION, 'access_secret')
        self.access_token = AccessToken(access_token, access_secret)

    @classmethod
    def load_from_path(cls, path):
        """Convenience method for loading credentials from a file.

        Open the file, create the Credentials and load from the file,
        and finally close the file and return the newly created
        Credentials instance.

        :param path: In which file the credential file should be saved.
        :type path: string
        :return: The loaded Credentials instance.
        :rtype: `Credentials`
        """
        credentials = cls()
        credentials_file = open(path, 'r')
        credentials.load(credentials_file)
        credentials_file.close()
        return credentials

    def save(self, writable_file):
        """Write the credentials to the file-like object.

        :param writable_file: A file-like object to write the credentials to
        :type writable_file: Any object supporting the file-like `write()`
            method
        :raise CredentialsFileError: when there is either no consumer or no
            access token
        """
        if self.consumer is None:
            raise CredentialsFileError('No consumer')
        if self.access_token is None:
            raise CredentialsFileError('No access token')
        parser = SafeConfigParser()
        parser.add_section(CREDENTIALS_FILE_VERSION)
        parser.set(CREDENTIALS_FILE_VERSION, 'consumer_key', self.consumer.key)
        parser.set(CREDENTIALS_FILE_VERSION, 'consumer_secret', self.consumer.secret)
        parser.set(CREDENTIALS_FILE_VERSION, 'access_token', self.access_token.key)
        parser.set(CREDENTIALS_FILE_VERSION, 'access_secret', self.access_token.secret)
        parser.write(writable_file)

    def save_to_path(self, path):
        """Convenience method for saving credentials to a file.

        Create the file, call self.save(), and close the
        file. Existing files are overwritten. The resulting file will
        be readable and writable only by the user.

        :param path: In which file the credential file should be saved.
        :type path: string
        """
        credentials_file = os.fdopen(os.open(path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, stat.S_IREAD | stat.S_IWRITE), 'w')
        self.save(credentials_file)
        credentials_file.close()

    def authorizeRequest(self, absolute_uri, method, body, headers):
        """Sign a request with OAuth credentials."""
        client = oauth1.Client(self.consumer.key, client_secret=self.consumer.secret, resource_owner_key=TruthyString(self.access_token.key or ''), resource_owner_secret=self.access_token.secret, signature_method=oauth1.SIGNATURE_PLAINTEXT, realm=self.oauth_realm)
        client.resource_owner_key = TruthyString(client.resource_owner_key)
        _, signed_headers, _ = client.sign(absolute_uri)
        for key, value in signed_headers.items():
            if six.PY2:
                key = key.encode('UTF-8')
                value = value.encode('UTF-8')
            headers[key] = value