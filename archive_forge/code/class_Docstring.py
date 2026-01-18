import enum
import typing as T
class Docstring:
    """Docstring object representation."""

    def __init__(self, style=None) -> None:
        """Initialize self."""
        self.short_description = None
        self.long_description = None
        self.blank_after_short_description = False
        self.blank_after_long_description = False
        self.meta = []
        self.style = style

    @property
    def description(self) -> T.Optional[str]:
        """Return the full description of the function

        Returns None if the docstring did not include any description
        """
        ret = []
        if self.short_description:
            ret.append(self.short_description)
            if self.blank_after_short_description:
                ret.append('')
        if self.long_description:
            ret.append(self.long_description)
        if not ret:
            return None
        return '\n'.join(ret)

    @property
    def params(self) -> T.List[DocstringParam]:
        """Return a list of information on function params."""
        return [item for item in self.meta if isinstance(item, DocstringParam)]

    @property
    def raises(self) -> T.List[DocstringRaises]:
        """Return a list of information on the exceptions that the function
        may raise.
        """
        return [item for item in self.meta if isinstance(item, DocstringRaises)]

    @property
    def returns(self) -> T.Optional[DocstringReturns]:
        """Return a single information on function return.

        Takes the first return information.
        """
        for item in self.meta:
            if isinstance(item, DocstringReturns):
                return item
        return None

    @property
    def many_returns(self) -> T.List[DocstringReturns]:
        """Return a list of information on function return."""
        return [item for item in self.meta if isinstance(item, DocstringReturns)]

    @property
    def deprecation(self) -> T.Optional[DocstringDeprecated]:
        """Return a single information on function deprecation notes."""
        for item in self.meta:
            if isinstance(item, DocstringDeprecated):
                return item
        return None

    @property
    def examples(self) -> T.List[DocstringExample]:
        """Return a list of information on function examples."""
        return [item for item in self.meta if isinstance(item, DocstringExample)]