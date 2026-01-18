import base64
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.twofactor import totp as crypto_totp
from oslo_log import log
from oslo_utils import timeutils
from keystone.auth import plugins
from keystone.auth.plugins import base
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _generate_totp_passcodes(secret, included_previous_windows=0):
    """Generate TOTP passcode.

    :param bytes secret: A base32 encoded secret for the TOTP authentication
    :returns: totp passcode as bytes
    """
    if isinstance(secret, str):
        secret = secret.encode('utf-8')
    while len(secret) % 8 != 0:
        secret = secret + b'='
    decoded = base64.b32decode(secret)
    totp = crypto_totp.TOTP(decoded, PASSCODE_LENGTH, hashes.SHA1(), PASSCODE_TIME_PERIOD, backend=default_backend())
    passcode_ts = timeutils.utcnow_ts(microsecond=True)
    passcodes = [totp.generate(passcode_ts).decode('utf-8')]
    for i in range(included_previous_windows):
        passcode_ts -= PASSCODE_TIME_PERIOD
        passcodes.append(totp.generate(passcode_ts).decode('utf-8'))
    return passcodes