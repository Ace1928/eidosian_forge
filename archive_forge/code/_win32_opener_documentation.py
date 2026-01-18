import errno
import time
import pywintypes
import win32con
import win32file
from oauth2client.contrib import locked_file
Close and unlock the file using the win32 primitive.