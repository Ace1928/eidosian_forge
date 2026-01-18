from boto.mashups.interactive import interactive_shell
import boto
import os
import time
import shutil
import paramiko
import socket
import subprocess
from boto.compat import StringIO

        Open a subprocess and run a command on the local host.
        
        :rtype: tuple
        :return: This function returns a tuple that contains an integer status
                and a string with the combined stdout and stderr output.
        