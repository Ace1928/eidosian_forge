import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def _authsvn(self, cmd, args=None):
    args = args and list(args) or []
    args.append(self._makeauthoptions())
    return self._svn(cmd, *args)