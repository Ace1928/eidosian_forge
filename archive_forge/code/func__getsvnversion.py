import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def _getsvnversion(ver=[]):
    try:
        return ver[0]
    except IndexError:
        v = py.process.cmdexec('svn -q --version')
        v.strip()
        v = '.'.join(v.split('.')[:2])
        ver.append(v)
        return v