import enum
import typing as T
class DocstringDeprecated(DocstringMeta):
    """DocstringMeta symbolizing deprecation metadata."""

    def __init__(self, args: T.List[str], description: T.Optional[str], version: T.Optional[str]) -> None:
        """Initialize self."""
        super().__init__(args, description)
        self.version = version
        self.description = description