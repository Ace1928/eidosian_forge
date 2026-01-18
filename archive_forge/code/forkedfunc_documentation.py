import py
import os
import sys
import marshal

    ForkedFunc provides a way to run a function in a forked process
    and get at its return value, stdout and stderr output as well
    as signals and exitstatusus.
