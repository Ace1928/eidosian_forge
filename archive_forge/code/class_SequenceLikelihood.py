from enum import Enum, Flag
class SequenceLikelihood:
    """
    This enum represents the likelihood of a character following the previous one.
    """
    NEGATIVE = 0
    UNLIKELY = 1
    LIKELY = 2
    POSITIVE = 3

    @classmethod
    def get_num_categories(cls) -> int:
        """:returns: The number of likelihood categories in the enum."""
        return 4