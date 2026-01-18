import os
import warnings
import builtins
import cherrypy
Warn if any socket_host is 'localhost'. See #711.