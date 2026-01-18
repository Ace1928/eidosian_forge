import hashlib
import socket
import time
from pyu2f import errors
from pyu2f import hardware
from pyu2f import hidtransport
from pyu2f import model
class U2FInterface(object):
    """High level U2F interface.

  Implements a high level interface in the spirit of the FIDO U2F
  javascript API high level interface.  It supports registration
  and authentication (signing).

  IMPORTANT NOTE: This class does NOT validate the app id against the
  origin.  In particular, any user can assert any app id all the way to
  the device.  The security model of a python library is such that doing
  so would not provide significant benfit as it could be bypassed by the
  caller talking to a lower level of the API.  In fact, so could the origin
  itself.  The origin is still set to a plausible value (the hostname) by
  this library.

  TODO(gdasher): Figure out a plan on how to address this gap/document the
  consequences of this more clearly.
  """

    def __init__(self, security_key, origin=socket.gethostname()):
        self.origin = origin
        self.security_key = security_key
        if self.security_key.CmdVersion() != b'U2F_V2':
            raise errors.UnsupportedVersionException()

    def Register(self, app_id, challenge, registered_keys):
        """Registers app_id with the security key.

    Executes the U2F registration flow with the security key.

    Args:
      app_id: The app_id to register the security key against.
      challenge: Server challenge passed to the security key.
      registered_keys: List of keys already registered for this app_id+user.

    Returns:
      RegisterResponse with key_handle and attestation information in it (
        encoded in FIDO U2F binary format within registration_data field).

    Raises:
      U2FError: There was some kind of problem with registration (e.g.
        the device was already registered or there was a timeout waiting
        for the test of user presence).
    """
        client_data = model.ClientData(model.ClientData.TYP_REGISTRATION, challenge, self.origin)
        challenge_param = self.InternalSHA256(client_data.GetJson())
        app_param = self.InternalSHA256(app_id)
        for key in registered_keys:
            try:
                if key.version != u'U2F_V2':
                    continue
                resp = self.security_key.CmdAuthenticate(challenge_param, app_param, key.key_handle, True)
                raise errors.HardwareError('Should Never Happen')
            except errors.TUPRequiredError:
                raise errors.U2FError(errors.U2FError.DEVICE_INELIGIBLE)
            except errors.InvalidKeyHandleError as e:
                pass
            except errors.HardwareError as e:
                raise errors.U2FError(errors.U2FError.BAD_REQUEST, e)
        for _ in range(30):
            try:
                resp = self.security_key.CmdRegister(challenge_param, app_param)
                return model.RegisterResponse(resp, client_data)
            except errors.TUPRequiredError as e:
                self.security_key.CmdWink()
                time.sleep(0.5)
            except errors.HardwareError as e:
                raise errors.U2FError(errors.U2FError.BAD_REQUEST, e)
        raise errors.U2FError(errors.U2FError.TIMEOUT)

    def Authenticate(self, app_id, challenge, registered_keys):
        """Authenticates app_id with the security key.

    Executes the U2F authentication/signature flow with the security key.

    Args:
      app_id: The app_id to register the security key against.
      challenge: Server challenge passed to the security key as a bytes object.
      registered_keys: List of keys already registered for this app_id+user.

    Returns:
      SignResponse with client_data, key_handle, and signature_data.  The client
      data is an object, while the signature_data is encoded in FIDO U2F binary
      format.

    Raises:
      U2FError: There was some kind of problem with authentication (e.g.
        there was a timeout while waiting for the test of user presence.)
    """
        client_data = model.ClientData(model.ClientData.TYP_AUTHENTICATION, challenge, self.origin)
        app_param = self.InternalSHA256(app_id)
        challenge_param = self.InternalSHA256(client_data.GetJson())
        num_invalid_keys = 0
        for key in registered_keys:
            try:
                if key.version != u'U2F_V2':
                    continue
                for _ in range(30):
                    try:
                        resp = self.security_key.CmdAuthenticate(challenge_param, app_param, key.key_handle)
                        return model.SignResponse(key.key_handle, resp, client_data)
                    except errors.TUPRequiredError:
                        self.security_key.CmdWink()
                        time.sleep(0.5)
            except errors.InvalidKeyHandleError:
                num_invalid_keys += 1
                continue
            except errors.HardwareError as e:
                raise errors.U2FError(errors.U2FError.BAD_REQUEST, e)
        if num_invalid_keys == len(registered_keys):
            raise errors.U2FError(errors.U2FError.DEVICE_INELIGIBLE)
        raise errors.U2FError(errors.U2FError.TIMEOUT)

    def InternalSHA256(self, string):
        md = hashlib.sha256()
        md.update(string.encode())
        return md.digest()