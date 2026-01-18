from lazyops.types.common import UpperStrEnum
from typing import Union
class UserRole(UpperStrEnum):
    ANON = 'ANON'
    READ = 'READ'
    WRITE = 'WRITE'
    MODIFY = 'MODIFY'
    PRIVILAGED = 'PRIVILAGED'
    USER = 'USER'
    USER_API = 'USER_API'
    USER_STAFF = 'USER_STAFF'
    USER_PRIVILAGED = 'USER_PRIVILAGED'
    USER_ADMIN = 'USER_ADMIN'
    STAFF = 'STAFF'
    SYSTEM = 'SYSTEM'
    SERVICE = 'SERVICE'
    API_CLIENT = 'API_CLIENT'
    ADMIN = 'ADMIN'
    SYSTEM_ADMIN = 'SYSTEM_ADMIN'
    SUPER_ADMIN = 'SUPER_ADMIN'

    @property
    def privilage_level(self) -> int:
        """
        Returns the privilage level of the user role
        - Can be subclassed to return a different value
        """
        return UserPrivilageLevel[self.value]

    def __eq__(self, other: Union[int, str, 'UserRole']) -> bool:
        """
        Returns True if the user role is equal to the other
        """
        if other is None:
            other = UserRole.ANON
        if isinstance(other, int):
            return self.privilage_level == other
        if hasattr(other, 'privilage_level'):
            return self.privilage_level == other.privilage_level
        return self.privilage_level == UserPrivilageLevel[other.upper()]

    def __ne__(self, other: Union[int, str, 'UserRole']) -> bool:
        """
        Returns True if the user role is not equal to the other
        """
        if other is None:
            other = UserRole.ANON
        if isinstance(other, int):
            return self.privilage_level != other
        if hasattr(other, 'privilage_level'):
            return self.privilage_level != other.privilage_level
        return self.privilage_level != UserPrivilageLevel[other.upper()]

    def __lt__(self, other: Union[int, str, 'UserRole']) -> bool:
        """
        Returns True if the user role is less than the other
        """
        if other is None:
            other = UserRole.ANON
        if isinstance(other, int):
            return self.privilage_level < other
        if hasattr(other, 'privilage_level'):
            return self.privilage_level < other.privilage_level
        return self.privilage_level < UserPrivilageLevel[other.upper()]

    def __le__(self, other: Union[int, 'UserRole']) -> bool:
        """
        Returns True if the user role is less than or equal to the other
        """
        if other is None:
            other = UserRole.ANON
        if isinstance(other, int):
            return self.privilage_level <= other
        if hasattr(other, 'privilage_level'):
            return self.privilage_level <= other.privilage_level
        return self.privilage_level <= UserPrivilageLevel[other.upper()]

    def __gt__(self, other: Union[int, str, 'UserRole']) -> bool:
        """
        Returns True if the user role is greater than the other
        """
        if other is None:
            other = UserRole.ANON
        if isinstance(other, int):
            return self.privilage_level > other
        if hasattr(other, 'privilage_level'):
            return self.privilage_level > other.privilage_level
        return self.privilage_level > UserPrivilageLevel[other.upper()]

    def __ge__(self, other: Union[int, str, 'UserRole']) -> bool:
        """
        Returns True if the user role is greater than or equal to the other
        """
        if other is None:
            other = UserRole.ANON
        if isinstance(other, int):
            return self.privilage_level >= other
        if hasattr(other, 'privilage_level'):
            return self.privilage_level >= other.privilage_level
        return self.privilage_level >= UserPrivilageLevel[other.upper()]

    def __contains__(self, other: Union[int, str, 'UserRole']) -> bool:
        """
        Returns True if the user role is contained in the other
        """
        if other is None:
            other = UserRole.ANON
        if isinstance(other, int):
            return self.privilage_level >= other
        if hasattr(other, 'privilage_level'):
            return self.privilage_level >= other.privilage_level
        return self.privilage_level >= UserPrivilageLevel[other.upper()]

    @classmethod
    def parse_role(cls, role: Union[str, int]) -> 'UserRole':
        """
        Parses the role
        """
        if role is None:
            return UserRole.ANON
        return cls(UserPrivilageIntLevel[role]) if isinstance(role, int) else cls(role.upper())

    def __hash__(self):
        """
        Returns the hash of the user role
        """
        return hash(self.value)