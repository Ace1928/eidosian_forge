import fcntl
import os
import struct
import subprocess
import sys
Emulates the most basic behavior of Linux's flock(1).