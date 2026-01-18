class InsertStrategy:
    """Indicates preferences on how to add multiple operations to a circuit."""
    NEW: 'InsertStrategy'
    NEW_THEN_INLINE: 'InsertStrategy'
    INLINE: 'InsertStrategy'
    EARLIEST: 'InsertStrategy'

    def __init__(self, name: str, doc: str):
        self.name = name
        self.__doc__ = doc

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'cirq.InsertStrategy.{self.name}'