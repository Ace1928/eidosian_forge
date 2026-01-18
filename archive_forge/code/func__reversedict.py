import re
def _reversedict(dict_to_reverse: dict) -> dict:
    """
    Internal helper for generating reverse mappings; given a
    dictionary, returns a new dictionary with keys and values swapped.

    """
    return {value: key for key, value in dict_to_reverse.items()}