importers when locating support scripts as well as when importing modules.
import sys
import importlib.machinery # importlib first so we can test #15386 via -m
import importlib.util
import io
import os
def _run_code(code, run_globals, init_globals=None, mod_name=None, mod_spec=None, pkg_name=None, script_name=None):
    """Helper to run code in nominated namespace"""
    if init_globals is not None:
        run_globals.update(init_globals)
    if mod_spec is None:
        loader = None
        fname = script_name
        cached = None
    else:
        loader = mod_spec.loader
        fname = mod_spec.origin
        cached = mod_spec.cached
        if pkg_name is None:
            pkg_name = mod_spec.parent
    run_globals.update(__name__=mod_name, __file__=fname, __cached__=cached, __doc__=None, __loader__=loader, __package__=pkg_name, __spec__=mod_spec)
    exec(code, run_globals)
    return run_globals