import base64
import datetime
import struct
import uuid
from cryptography import fernet
import msgpack
from oslo_log import log
from oslo_utils import timeutils
from keystone.auth import plugins as auth_plugins
from keystone.common import fernet_utils as utils
from keystone.common import utils as ks_utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
class TokenFormatter(object):
    """Packs and unpacks payloads into tokens for transport."""

    @property
    def crypto(self):
        """Return a cryptography instance.

        You can extend this class with a custom crypto @property to provide
        your own token encoding / decoding. For example, using a different
        cryptography library (e.g. ``python-keyczar``) or to meet arbitrary
        security requirements.

        This @property just needs to return an object that implements
        ``encrypt(plaintext)`` and ``decrypt(ciphertext)``.

        """
        fernet_utils = utils.FernetUtils(CONF.fernet_tokens.key_repository, CONF.fernet_tokens.max_active_keys, 'fernet_tokens')
        keys = fernet_utils.load_keys()
        if not keys:
            raise exception.KeysNotFound()
        fernet_instances = [fernet.Fernet(key) for key in keys]
        return fernet.MultiFernet(fernet_instances)

    def pack(self, payload):
        """Pack a payload for transport as a token.

        :type payload: bytes
        :rtype: str

        """
        return self.crypto.encrypt(payload).rstrip(b'=').decode('utf-8')

    def unpack(self, token):
        """Unpack a token, and validate the payload.

        :type token: str
        :rtype: bytes

        """
        token = TokenFormatter.restore_padding(token)
        try:
            return self.crypto.decrypt(token.encode('utf-8'))
        except fernet.InvalidToken:
            raise exception.ValidationError(_('Could not recognize Fernet token'))

    @classmethod
    def restore_padding(cls, token):
        """Restore padding based on token size.

        :param token: token to restore padding on
        :type token: str
        :returns: token with correct padding

        """
        mod_returned = len(token) % 4
        if mod_returned:
            missing_padding = 4 - mod_returned
            token += '=' * missing_padding
        return token

    @classmethod
    def creation_time(cls, fernet_token):
        """Return the creation time of a valid Fernet token.

        :type fernet_token: str

        """
        fernet_token = TokenFormatter.restore_padding(fernet_token)
        token_bytes = base64.urlsafe_b64decode(fernet_token.encode('utf-8'))
        timestamp_bytes = token_bytes[TIMESTAMP_START:TIMESTAMP_END]
        timestamp_int = struct.unpack('>Q', timestamp_bytes)[0]
        issued_at = datetime.datetime.utcfromtimestamp(timestamp_int)
        return issued_at

    def create_token(self, user_id, expires_at, audit_ids, payload_class, methods=None, system=None, domain_id=None, project_id=None, trust_id=None, federated_group_ids=None, identity_provider_id=None, protocol_id=None, access_token_id=None, app_cred_id=None, thumbprint=None):
        """Given a set of payload attributes, generate a Fernet token."""
        version = payload_class.version
        payload = payload_class.assemble(user_id, methods, system, project_id, domain_id, expires_at, audit_ids, trust_id, federated_group_ids, identity_provider_id, protocol_id, access_token_id, app_cred_id, thumbprint)
        versioned_payload = (version,) + payload
        serialized_payload = msgpack.packb(versioned_payload)
        token = self.pack(serialized_payload)
        if len(token) > CONF.max_token_size:
            LOG.info(f'Fernet token created with length of {len(token)} characters, which exceeds {CONF.max_token_size} characters')
        return token

    def validate_token(self, token):
        """Validate a Fernet token and returns the payload attributes.

        :type token: str

        """
        serialized_payload = self.unpack(token)
        try:
            versioned_payload = msgpack.unpackb(serialized_payload)
        except UnicodeDecodeError:
            versioned_payload = msgpack.unpackb(serialized_payload, raw=True)
        version, payload = (versioned_payload[0], versioned_payload[1:])
        for payload_class in _PAYLOAD_CLASSES:
            if version == payload_class.version:
                user_id, methods, system, project_id, domain_id, expires_at, audit_ids, trust_id, federated_group_ids, identity_provider_id, protocol_id, access_token_id, app_cred_id, thumbprint = payload_class.disassemble(payload)
                break
        else:
            raise exception.ValidationError(_('This is not a recognized Fernet payload version: %s') % version)
        if isinstance(system, bytes):
            system = system.decode('utf-8')
        issued_at = TokenFormatter.creation_time(token)
        issued_at = ks_utils.isotime(at=issued_at, subsecond=True)
        expires_at = timeutils.parse_isotime(expires_at)
        expires_at = ks_utils.isotime(at=expires_at, subsecond=True)
        return (user_id, methods, audit_ids, system, domain_id, project_id, trust_id, federated_group_ids, identity_provider_id, protocol_id, access_token_id, app_cred_id, thumbprint, issued_at, expires_at)