from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import threading
class ManifestMessage(ThreadMessage):
    """Message class for updating manifest file with result of copy.

  Attributes:
    source_url (StorageUrl): Source URL. Used to match information recorded by
      copy progress infra (FilesAndBytesStatusTracker).
    destination_url (StorageUrl): Destination URL.
    end_time (datetime): Date and time copy completed.
    size (int): Size of file or object.
    result_status (manifest_utils.ResultStatus): End status of copy. Either
      "OK", "skip", or "error".
    md5_hash (str|None): Hash of copied file or object.
    description (str|None): Message about something that happened during a copy.
  """

    def __init__(self, source_url, destination_url, end_time, size, result_status, md5_hash=None, description=None):
        """Initializes ManifestMessage. Args in attributes docstring."""
        self.source_url = source_url
        self.destination_url = destination_url
        self.end_time = end_time
        self.size = size
        self.result_status = result_status
        self.md5_hash = md5_hash
        self.description = description

    def __repr__(self):
        """Returns a string with a valid constructor for this message."""
        source_url_string = "'{}'".format(self.source_url)
        destination_url_string = "'{}'".format(self.destination_url)
        end_time_string = "'{}'".format(self.end_time)
        md5_hash_string = "'{}'".format(self.md5_hash) if self.md5_hash else 'None'
        description_string = "'{}'".format(self.description) if self.description else 'None'
        return '{class_name}(source_url={source_url}, destination_url={destination_url}, end_time={end_time}, size={size}, result_status={result_status}, md5_hash={md5_hash}, description={description})'.format(class_name=self.__class__.__name__, source_url=source_url_string, destination_url=destination_url_string, end_time=end_time_string, size=self.size, result_status=self.result_status, md5_hash=md5_hash_string, description=description_string)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.source_url == other.source_url and self.destination_url == other.destination_url and (self.end_time == other.end_time) and (self.size == other.size) and (self.result_status == other.result_status) and (self.md5_hash == other.md5_hash) and (self.description == other.description)