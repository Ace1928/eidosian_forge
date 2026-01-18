from typing import Any, Type, Optional, Set, Dict
class DaciteFieldError(DaciteError):

    def __init__(self, field_path: Optional[str]=None):
        super().__init__()
        self.field_path = field_path

    def update_path(self, parent_field_path: str) -> None:
        if self.field_path:
            self.field_path = f'{parent_field_path}.{self.field_path}'
        else:
            self.field_path = parent_field_path