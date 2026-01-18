from __future__ import annotations
class NeighborsNotComputedChemenvError(AbstractChemenvError):
    """Neighbors not computed error."""

    def __init__(self, site):
        """
        Args:
            site:
        """
        self.site = site

    def __str__(self):
        return f'The neighbors were not computed for the following site : \n{self.site}'