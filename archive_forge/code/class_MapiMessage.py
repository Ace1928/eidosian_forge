import os
from ctypes import *
class MapiMessage(Structure):
    _fields_ = [('ulReserved', c_ulong), ('lpszSubject', c_char_p), ('lpszNoteText', c_char_p), ('lpszMessageType', c_char_p), ('lpszDateReceived', c_char_p), ('lpszConversationID', c_char_p), ('flFlags', FLAGS), ('lpOriginator', lpMapiRecipDesc), ('nRecipCount', c_ulong), ('lpRecips', lpMapiRecipDesc), ('nFileCount', c_ulong), ('lpFiles', lpMapiFileDesc)]