import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def _GetDeps(self, dependencies, visited=None):
    """Recursively finds dependencies for file protos.

    Args:
      dependencies: The names of the files being depended on.
      visited: The names of files already found.

    Yields:
      Each direct and indirect dependency.
    """
    visited = visited or set()
    for dependency in dependencies:
        if dependency not in visited:
            visited.add(dependency)
            dep_desc = self.FindFileByName(dependency)
            yield dep_desc
            public_files = [d.name for d in dep_desc.public_dependencies]
            yield from self._GetDeps(public_files, visited)