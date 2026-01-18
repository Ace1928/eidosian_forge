import enum
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
class UploadType(enum.Enum):
    """Enum class for specifying upload type for diagnostic tests."""
    PARALLEL_COMPOSITE = 'PARALLEL_COMPOSITE'
    STREAMING = 'STREAMING'
    FILE = 'FILE'