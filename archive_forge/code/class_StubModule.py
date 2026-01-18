from __future__ import annotations
import argparse
import os
import typing as t
class StubModule:

    def __getattr__(self, attr: str) -> t.Any:
        if not attr.startswith('__'):
            raise ModuleNotFoundError("No module named 'argcomplete'")
        raise AttributeError(f"argcomplete stub module has no attribute '{attr}'")