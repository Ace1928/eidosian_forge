import os
import win32api
import win32con
import win32event
import win32service
import win32serviceutil
from cherrypy.process import wspbus, plugins
def SvcStop(self):
    from cherrypy import process
    self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
    process.bus.exit()