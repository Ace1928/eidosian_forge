import enum
import typing as T
class DocstringParam(DocstringMeta):
    """DocstringMeta symbolizing :param metadata."""

    def __init__(self, args: T.List[str], description: T.Optional[str], arg_name: str, type_name: T.Optional[str], is_optional: T.Optional[bool], default: T.Optional[str]) -> None:
        """Initialize self."""
        super().__init__(args, description)
        self.arg_name = arg_name
        self.type_name = type_name
        self.is_optional = is_optional
        self.default = default