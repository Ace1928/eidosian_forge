from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2KernelRootkit(_messages.Message):
    """Kernel mode rootkit signatures.

  Fields:
    name: Rootkit name, when available.
    unexpectedCodeModification: True if unexpected modifications of kernel
      code memory are present.
    unexpectedFtraceHandler: True if `ftrace` points are present with
      callbacks pointing to regions that are not in the expected kernel or
      module code range.
    unexpectedInterruptHandler: True if interrupt handlers that are are not in
      the expected kernel or module code regions are present.
    unexpectedKernelCodePages: True if kernel code pages that are not in the
      expected kernel or module code regions are present.
    unexpectedKprobeHandler: True if `kprobe` points are present with
      callbacks pointing to regions that are not in the expected kernel or
      module code range.
    unexpectedProcessesInRunqueue: True if unexpected processes in the
      scheduler run queue are present. Such processes are in the run queue,
      but not in the process task list.
    unexpectedReadOnlyDataModification: True if unexpected modifications of
      kernel read-only data memory are present.
    unexpectedSystemCallHandler: True if system call handlers that are are not
      in the expected kernel or module code regions are present.
  """
    name = _messages.StringField(1)
    unexpectedCodeModification = _messages.BooleanField(2)
    unexpectedFtraceHandler = _messages.BooleanField(3)
    unexpectedInterruptHandler = _messages.BooleanField(4)
    unexpectedKernelCodePages = _messages.BooleanField(5)
    unexpectedKprobeHandler = _messages.BooleanField(6)
    unexpectedProcessesInRunqueue = _messages.BooleanField(7)
    unexpectedReadOnlyDataModification = _messages.BooleanField(8)
    unexpectedSystemCallHandler = _messages.BooleanField(9)