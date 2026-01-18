import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def FindFileByName(self, file_name):
    """Gets a FileDescriptor by file name.

    Args:
      file_name (str): The path to the file to get a descriptor for.

    Returns:
      FileDescriptor: The descriptor for the named file.

    Raises:
      KeyError: if the file cannot be found in the pool.
    """
    try:
        return self._file_descriptors[file_name]
    except KeyError:
        pass
    try:
        file_proto = self._internal_db.FindFileByName(file_name)
    except KeyError as error:
        if self._descriptor_db:
            file_proto = self._descriptor_db.FindFileByName(file_name)
        else:
            raise error
    if not file_proto:
        raise KeyError('Cannot find a file named %s' % file_name)
    return self._ConvertFileProtoToFileDescriptor(file_proto)