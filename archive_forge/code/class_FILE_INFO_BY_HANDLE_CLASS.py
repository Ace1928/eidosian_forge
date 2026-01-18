import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class FILE_INFO_BY_HANDLE_CLASS(object):
    FileBasicInfo = 0
    FileStandardInfo = 1
    FileNameInfo = 2
    FileRenameInfo = 3
    FileDispositionInfo = 4
    FileAllocationInfo = 5
    FileEndOfFileInfo = 6
    FileStreamInfo = 7
    FileCompressionInfo = 8
    FileAttributeTagInfo = 9
    FileIdBothDirectoryInfo = 10
    FileIdBothDirectoryRestartInfo = 11
    FileIoPriorityHintInfo = 12
    MaximumFileInfoByHandlesClass = 13