import copy
import importlib.metadata as importlib_metadata
import importlib.util
import weakref
from functools import partialmethod
from ..dependency_versions_check import dep_version_check
from ..utils import is_accelerate_available, is_torch_available, logging

        This stage is run after we have the model and know num_training_steps.

        Now we can complete the configuration process.
        