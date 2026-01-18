import dataclasses
from dataclasses import field
from types import CodeType, ModuleType
from typing import Any, Dict
def _add_mod(self, mod):
    if mod.__name__ not in self.name_to_modrec:
        self.name_to_modrec[mod.__name__] = ModuleRecord(mod)
    return self.name_to_modrec[mod.__name__]