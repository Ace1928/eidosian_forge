class NotebookFloatFlag:
    """Stub for absl flag to be used within a jupyter notebook."""

    def __init__(self, name: str, value: float, doc: str):
        self.__name = name
        self.__value = value
        self.__doc__ = doc

    @property
    def value(self) -> float:
        """Returns the value passed at creation."""
        return self.__value

    @property
    def name(self) -> str:
        """Returns the name of the parameter."""
        return self.__name