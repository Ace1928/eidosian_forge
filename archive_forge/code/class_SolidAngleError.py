from __future__ import annotations
class SolidAngleError(AbstractChemenvError):
    """Solid angle error."""

    def __init__(self, cosinus):
        """
        Args:
            cosinus:
        """
        self.cosinus = cosinus

    def __str__(self):
        return f'Value of cosinus ({self.cosinus}) from which an angle should be retrieved is not between -1.0 and 1.0'