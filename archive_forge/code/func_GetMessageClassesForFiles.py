import warnings
from google.protobuf.internal import api_implementation
from google.protobuf import descriptor_pool
from google.protobuf import message
def GetMessageClassesForFiles(files, pool):
    """Gets all the messages from specified files.

  This will find and resolve dependencies, failing if the descriptor
  pool cannot satisfy them.

  Args:
    files: The file names to extract messages from.
    pool: The descriptor pool to find the files including the dependent
      files.

  Returns:
    A dictionary mapping proto names to the message classes.
  """
    result = {}
    for file_name in files:
        file_desc = pool.FindFileByName(file_name)
        for desc in file_desc.message_types_by_name.values():
            result[desc.full_name] = GetMessageClass(desc)
        for extension in file_desc.extensions_by_name.values():
            extended_class = GetMessageClass(extension.containing_type)
            if api_implementation.Type() != 'python':
                if extension is not pool.FindExtensionByNumber(extension.containing_type, extension.number):
                    raise ValueError('Double registration of Extensions')
            if extension.message_type:
                GetMessageClass(extension.message_type)
    return result