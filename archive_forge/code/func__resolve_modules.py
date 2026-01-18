import dataclasses
from dataclasses import field
from types import CodeType, ModuleType
from typing import Any, Dict
@classmethod
def _resolve_modules(cls, vars):

    def resolve_module(var):
        if not isinstance(var, ModuleRecord):
            return var
        dummy_mod = DummyModule(var.module.__name__)
        for attr_name, attr_value in var.accessed_attrs.items():
            attr_value = resolve_module(attr_value)
            dummy_mod.__setattr__(attr_name, attr_value)
        return dummy_mod
    return {k: resolve_module(v) for k, v in vars.items()}