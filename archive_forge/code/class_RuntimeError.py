import builtins
from typing import Type
class RuntimeError(builtins.RuntimeError, CryptoError):
    pass