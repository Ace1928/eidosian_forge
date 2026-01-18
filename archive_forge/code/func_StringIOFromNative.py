import io
import os
import sys
import unittest
import cherrypy
from cherrypy.test import helper
def StringIOFromNative(x):
    return io.StringIO(str(x))