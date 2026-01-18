import logging; log = logging.getLogger(__name__)
import sys
from passlib.utils.decor import deprecated_method
from abc import ABCMeta, abstractmethod, abstractproperty
class DisabledHash(PasswordHash):
    """
    extended disabled-hash methods; only need be present if .disabled = True
    """
    is_disabled = True

    @classmethod
    def disable(cls, hash=None):
        """
        return string representing a 'disabled' hash;
        optionally including previously enabled hash
        (this is up to the individual scheme).
        """
        return cls.hash('')

    @classmethod
    def enable(cls, hash):
        """
        given a disabled-hash string,
        extract previously-enabled hash if one is present,
        otherwise raises ValueError
        """
        raise ValueError('cannot restore original hash')