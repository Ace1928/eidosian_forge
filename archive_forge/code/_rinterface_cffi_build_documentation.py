import cffi  # type: ignore
import os
import re
import sys
import warnings
import situation  # preloaded in setup.py
import importlib

    Rudimentary C-preprocessor for ifdef blocks.

    Args:
    - csource: iterator C source code
    - definitions: a mapping (e.g., set or dict contaning
      which "names" are defined)

    Returns:
    The csource with the conditional ifdef blocks for name
    processed.
    