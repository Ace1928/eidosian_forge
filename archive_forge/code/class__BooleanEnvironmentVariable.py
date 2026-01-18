import os
from pathlib import Path
class _BooleanEnvironmentVariable(_EnvironmentVariable):
    """
    Represents a boolean environment variable.
    """

    def __init__(self, name, default):
        if not (default is True or default is False or default is None):
            raise ValueError(f'{name} default value must be one of [True, False, None]')
        super().__init__(name, bool, default)

    def get(self):
        if not self.defined:
            return self.default
        val = os.getenv(self.name)
        lowercased = val.lower()
        if lowercased not in ['true', 'false', '1', '0']:
            raise ValueError(f"{self.name} value must be one of ['true', 'false', '1', '0'] (case-insensitive), but got {val}")
        return lowercased in ['true', '1']