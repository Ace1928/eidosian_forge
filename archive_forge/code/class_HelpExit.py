import argparse
import inspect
import traceback
import autopage.argparse
from . import command
class HelpExit(SystemExit):
    """Special exception type to trigger quick exit from the application

    We subclass from SystemExit to preserve API compatibility for
    anything that used to catch SystemExit, but use a different class
    so that cliff's Application can tell the difference between
    something trying to hard-exit and help saying it's done.
    """