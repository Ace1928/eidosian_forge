class WrongPasswordError(FileNotDecryptedError):
    """Raised when the wrong password is used to try to decrypt an encrypted PDF file."""