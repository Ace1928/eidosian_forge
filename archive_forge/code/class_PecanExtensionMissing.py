import sys
import pkg_resources
import inspect
import logging
class PecanExtensionMissing(ImportError):
    pass