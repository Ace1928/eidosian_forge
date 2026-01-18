import time
from pydantic import BaseModel, Field
from typing import Optional, Union, List, Dict
class UserJWTClaims(BaseClaims):
    """
    User JWT Claims
    """
    exp: int
    iat: int

    @property
    def is_expired(self) -> bool:
        """
        Returns True if the Token is Expired
        """
        return self.exp < int(time.time()) - 45

    def to_api_key_claims(self) -> 'APIKeyJWTClaims':
        """
        Converts the User JWT Claims to an API Key JWT Claims
        """
        return APIKeyJWTClaims.model_validate(self, from_attributes=True)