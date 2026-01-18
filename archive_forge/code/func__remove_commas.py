import re
def _remove_commas(self, m: str) -> str:
    """
        This method is used to remove commas from sentences.
        """
    return m.group(1).replace(',', '')