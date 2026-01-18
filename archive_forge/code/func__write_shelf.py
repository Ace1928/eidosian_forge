import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
@classmethod
def _write_shelf(cls, shelf_file, transform, revision_id, message=None):
    serializer = pack.ContainerSerialiser()
    shelf_file.write(serializer.begin())
    metadata = cls.metadata_record(serializer, revision_id, message)
    shelf_file.write(metadata)
    for bytes in transform.serialize(serializer):
        shelf_file.write(bytes)
    shelf_file.write(serializer.end())