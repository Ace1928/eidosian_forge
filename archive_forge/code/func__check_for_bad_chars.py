import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def _check_for_bad_chars(text, allowed_chars=ALLOWED_CHARS):
    for c in str(text):
        if c.isalnum():
            continue
        if c in allowed_chars:
            continue
        return True
    return False