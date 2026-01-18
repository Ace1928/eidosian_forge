from __future__ import annotations
import abc
import logging
import signal
import typing
class ExitMainLoop(Exception):
    """
    When this exception is raised within a main loop the main loop
    will exit cleanly.
    """