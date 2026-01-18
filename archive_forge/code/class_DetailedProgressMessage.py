from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import threading
class DetailedProgressMessage(ThreadMessage):
    """Message class for sending information about operation progress.

  This class contains specific information on the progress of operating on a
  file, cloud object, or single component.

  Attributes:
    offset (int): Start of byte range to start operation at.
    length (int): Total size of file or component in bytes.
    current_byte (int): Index of byte being operated on.
    finished (bool): Indicates if the operation is complete.
    time (float): When message was created (seconds since epoch).
    source_url (StorageUrl): Represents source of data used by operation.
    destination_url (StorageUrl|None): Represents destination of data used by
      operation. None for unary operations like hashing.
    component_number (int|None): If a multipart operation, indicates the
      component number.
    total_components (int|None): If a multipart operation, indicates the
      total number of components.
    operation_name (task_status.OperationName|None): Name of the operation
      running on target data.
    process_id (int|None): Identifies process that produced the instance of this
      message (overridable for testing).
    thread_id (int|None): Identifies thread that produced the instance of this
      message (overridable for testing).
  """

    def __init__(self, offset, length, current_byte, time, source_url, destination_url=None, component_number=None, total_components=None, operation_name=None, process_id=None, thread_id=None):
        """Initializes a ProgressMessage. See attributes docstring for arguments."""
        self.offset = offset
        self.length = length
        self.current_byte = current_byte
        self.time = time
        self.source_url = source_url
        self.destination_url = destination_url
        self.component_number = component_number
        self.total_components = total_components
        self.operation_name = operation_name
        self.process_id = process_id or os.getpid()
        self.thread_id = thread_id or threading.current_thread().ident

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __repr__(self):
        """Returns a string with a valid constructor for this message."""
        source_url_string = "'{}'".format(self.source_url)
        destination_url_string = "'{}'".format(self.destination_url) if self.destination_url else None
        operation_name_string = "'{}'".format(self.operation_name.value) if self.operation_name else None
        return '{class_name}(time={time}, offset={offset}, length={length}, current_byte={current_byte}, source_url={source_url}, destination_url={destination_url}, component_number={component_number}, total_components={total_components}, operation_name={operation_name}, process_id={process_id}, thread_id={thread_id})'.format(class_name=self.__class__.__name__, time=self.time, offset=self.offset, length=self.length, current_byte=self.current_byte, source_url=source_url_string, destination_url=destination_url_string, component_number=self.component_number, total_components=self.total_components, operation_name=operation_name_string, process_id=self.process_id, thread_id=self.thread_id)