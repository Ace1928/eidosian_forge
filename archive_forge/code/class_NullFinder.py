import sys
import platform
class NullFinder:
    """
    A "Finder" (aka "MetaClassFinder") that never finds any modules,
    but may find distributions.
    """

    @staticmethod
    def find_spec(*args, **kwargs):
        return None
    find_module = find_spec