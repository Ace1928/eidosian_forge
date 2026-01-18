import os
import sys
import warnings
def _SetType(implementation_type):
    """Never use! Only for protobuf benchmark."""
    global _implementation_type
    _implementation_type = implementation_type