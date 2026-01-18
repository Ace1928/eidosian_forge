from os import environ
import shlex
import subprocess
import sys
import pytest
def _kivy_subproces_import(env):
    return subprocess.run([sys.executable, '-c', 'import kivy', '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env).stdout.decode('utf8')