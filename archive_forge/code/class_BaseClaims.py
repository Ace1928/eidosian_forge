import time
from pydantic import BaseModel, Field
from typing import Optional, Union, List, Dict
class BaseClaims(BaseModel):
    """
    Base JWT Claims
    """
    aud: Union[str, List[str]]
    azp: str
    iss: str
    scope: str = ''
    sub: str
    cust_user_id_perf_test: Optional[str] = None
    user_roles: Optional[List[str]] = Field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        """
        Returns True if the Token is Expired
        """
        return False

    @property
    def scopes(self) -> List[str]:
        """
        Returns the Scopes
        """
        return self.scope.split(' ')