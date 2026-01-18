from typing import Any, Type, Optional, Set, Dict
class MissingValueError(DaciteFieldError):

    def __init__(self, field_path: Optional[str]=None):
        super().__init__(field_path=field_path)

    def __str__(self) -> str:
        return f'missing value for field "{self.field_path}"'