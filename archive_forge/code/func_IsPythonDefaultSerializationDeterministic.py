import importlib
import os
import sys
import warnings
def IsPythonDefaultSerializationDeterministic():
    return _python_deterministic_proto_serialization