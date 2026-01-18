from abc import ABC, abstractmethod
from .header import Field
class DataWarning(Warning):
    """Base class for warnings about tractogram file data."""