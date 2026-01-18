from cursive.i18n import _
class SignatureVerificationError(CursiveException):
    msg_fmt = _('Signature verification for the image failed: %(reason)s.')