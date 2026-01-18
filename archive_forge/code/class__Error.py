importers when locating support scripts as well as when importing modules.
import sys
import importlib.machinery # importlib first so we can test #15386 via -m
import importlib.util
import io
import os
class _Error(Exception):
    """Error that _run_module_as_main() should report without a traceback"""