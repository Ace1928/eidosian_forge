import hashlib
import os
import random
from operator import attrgetter
import gyp.common
class MSVSFolder(MSVSSolutionEntry):
    """Folder in a Visual Studio project or solution."""

    def __init__(self, path, name=None, entries=None, guid=None, items=None):
        """Initializes the folder.

    Args:
      path: Full path to the folder.
      name: Name of the folder.
      entries: List of folder entries to nest inside this folder.  May contain
          Folder or Project objects.  May be None, if the folder is empty.
      guid: GUID to use for folder, if not None.
      items: List of solution items to include in the folder project.  May be
          None, if the folder does not directly contain items.
    """
        if name:
            self.name = name
        else:
            self.name = os.path.basename(path)
        self.path = path
        self.guid = guid
        self.entries = sorted(entries or [], key=attrgetter('path'))
        self.items = list(items or [])
        self.entry_type_guid = ENTRY_TYPE_GUIDS['folder']

    def get_guid(self):
        if self.guid is None:
            self.guid = MakeGuid(self.path, seed='msvs_folder')
        return self.guid