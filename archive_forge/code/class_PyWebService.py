import os
import win32api
import win32con
import win32event
import win32service
import win32serviceutil
from cherrypy.process import wspbus, plugins
class PyWebService(win32serviceutil.ServiceFramework):
    """Python Web Service."""
    _svc_name_ = 'Python Web Service'
    _svc_display_name_ = 'Python Web Service'
    _svc_deps_ = None
    _exe_name_ = 'pywebsvc'
    _exe_args_ = None
    _svc_description_ = 'Python Web Service'

    def SvcDoRun(self):
        from cherrypy import process
        process.bus.start()
        process.bus.block()

    def SvcStop(self):
        from cherrypy import process
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        process.bus.exit()

    def SvcOther(self, control):
        from cherrypy import process
        process.bus.publish(control_codes.key_for(control))