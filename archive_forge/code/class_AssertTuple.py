class AssertTuple(Message):
    """
    Assertion test is a non-empty tuple literal, which are always True.
    """
    message = 'assertion is always true, perhaps remove parentheses?'