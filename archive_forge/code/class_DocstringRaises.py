import enum
import typing as T
class DocstringRaises(DocstringMeta):
    """DocstringMeta symbolizing :raises metadata."""

    def __init__(self, args: T.List[str], description: T.Optional[str], type_name: T.Optional[str]) -> None:
        """Initialize self."""
        super().__init__(args, description)
        self.type_name = type_name
        self.description = description