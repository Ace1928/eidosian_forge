import os
import win32api
import win32con
import win32event
import win32service
import win32serviceutil
from cherrypy.process import wspbus, plugins
def SvcDoRun(self):
    from cherrypy import process
    process.bus.start()
    process.bus.block()