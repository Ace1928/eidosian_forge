from typing import Any, Type, Optional, Set, Dict
class StrictUnionMatchError(DaciteFieldError):

    def __init__(self, union_matches: Dict[Type, Any], field_path: Optional[str]=None) -> None:
        super().__init__(field_path=field_path)
        self.union_matches = union_matches

    def __str__(self) -> str:
        conflicting_types = ', '.join((_name(type_) for type_ in self.union_matches))
        return f'can not choose between possible Union matches for field "{self.field_path}": {conflicting_types}'