from typing import Any, Callable, List, Optional, Sequence
from fastapi._compat import ModelField
from fastapi.security.base import SecurityBase
class SecurityRequirement:

    def __init__(self, security_scheme: SecurityBase, scopes: Optional[Sequence[str]]=None):
        self.security_scheme = security_scheme
        self.scopes = scopes