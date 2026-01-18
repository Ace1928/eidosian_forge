class IfTuple(Message):
    """
    Conditional test is a non-empty tuple literal, which are always True.
    """
    message = "'if tuple literal' is always true, perhaps remove accidental comma?"