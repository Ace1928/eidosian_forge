import os
import abc
class Credential(metaclass=abc.ABCMeta):
    """Abstract class to manage credentials"""

    @abc.abstractproperty
    def username(self):
        return None

    @abc.abstractproperty
    def password(self):
        return None