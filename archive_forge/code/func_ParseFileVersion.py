from typing import Optional
from tensorboard.compat.proto import event_pb2
from tensorboard.util import tb_logging
def ParseFileVersion(file_version: str) -> float:
    """Convert the string file_version in event.proto into a float.

    Args:
      file_version: String file_version from event.proto

    Returns:
      Version number as a float.
    """
    tokens = file_version.split('brain.Event:')
    try:
        return float(tokens[-1])
    except ValueError:
        logger.warning('Invalid event.proto file_version. Defaulting to use of out-of-order event.step logic for purging expired events.')
        return -1