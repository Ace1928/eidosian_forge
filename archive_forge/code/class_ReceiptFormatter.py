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
class ReceiptFormatter(object):
    """Packs and unpacks payloads into receipts for transport."""

    @property
    def crypto(self):
        """Return a cryptography instance.

        You can extend this class with a custom crypto @property to provide
        your own receipt encoding / decoding. For example, using a different
        cryptography library (e.g. ``python-keyczar``) or to meet arbitrary
        security requirements.

        This @property just needs to return an object that implements
        ``encrypt(plaintext)`` and ``decrypt(ciphertext)``.

        """
        fernet_utils = utils.FernetUtils(CONF.fernet_receipts.key_repository, CONF.fernet_receipts.max_active_keys, 'fernet_receipts')
        keys = fernet_utils.load_keys()
        if not keys:
            raise exception.KeysNotFound()
        fernet_instances = [fernet.Fernet(key) for key in keys]
        return fernet.MultiFernet(fernet_instances)

    def pack(self, payload):
        """Pack a payload for transport as a receipt.

        :type payload: bytes
        :rtype: str

        """
        return self.crypto.encrypt(payload).rstrip(b'=').decode('utf-8')

    def unpack(self, receipt):
        """Unpack a receipt, and validate the payload.

        :type receipt: str
        :rtype: bytes

        """
        receipt = ReceiptFormatter.restore_padding(receipt)
        try:
            return self.crypto.decrypt(receipt.encode('utf-8'))
        except fernet.InvalidToken:
            raise exception.ValidationError(_('This is not a recognized Fernet receipt %s') % receipt)

    @classmethod
    def restore_padding(cls, receipt):
        """Restore padding based on receipt size.

        :param receipt: receipt to restore padding on
        :type receipt: str
        :returns: receipt with correct padding

        """
        mod_returned = len(receipt) % 4
        if mod_returned:
            missing_padding = 4 - mod_returned
            receipt += '=' * missing_padding
        return receipt

    @classmethod
    def creation_time(cls, fernet_receipt):
        """Return the creation time of a valid Fernet receipt.

        :type fernet_receipt: str

        """
        fernet_receipt = ReceiptFormatter.restore_padding(fernet_receipt)
        receipt_bytes = base64.urlsafe_b64decode(fernet_receipt.encode('utf-8'))
        timestamp_bytes = receipt_bytes[TIMESTAMP_START:TIMESTAMP_END]
        timestamp_int = struct.unpack('>Q', timestamp_bytes)[0]
        issued_at = datetime.datetime.utcfromtimestamp(timestamp_int)
        return issued_at

    def create_receipt(self, user_id, methods, expires_at):
        """Given a set of payload attributes, generate a Fernet receipt."""
        payload = ReceiptPayload.assemble(user_id, methods, expires_at)
        serialized_payload = msgpack.packb(payload)
        receipt = self.pack(serialized_payload)
        if len(receipt) > 255:
            LOG.info('Fernet receipt created with length of %d characters, which exceeds 255 characters', len(receipt))
        return receipt

    def validate_receipt(self, receipt):
        """Validate a Fernet receipt and returns the payload attributes.

        :type receipt: str

        """
        serialized_payload = self.unpack(receipt)
        payload = msgpack.unpackb(serialized_payload)
        user_id, methods, expires_at = ReceiptPayload.disassemble(payload)
        issued_at = ReceiptFormatter.creation_time(receipt)
        issued_at = ks_utils.isotime(at=issued_at, subsecond=True)
        expires_at = timeutils.parse_isotime(expires_at)
        expires_at = ks_utils.isotime(at=expires_at, subsecond=True)
        return (user_id, methods, issued_at, expires_at)