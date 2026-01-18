from __future__ import annotations
__import__ wrapper - does not change imports at all, but tracks them.

        Default order is implemented by doing output directly.
        All other orders are implemented by collecting output information into
        a sorted list that will be emitted after all imports are processed.

        Indirect imports can only occur after the requested symbol has been
        imported directly (because the indirect import would not have a module
        to pick the symbol up from).
        So this code detects indirect imports by checking whether the symbol in
        question was already imported.

        Keeps the semantics of __import__ unchanged.