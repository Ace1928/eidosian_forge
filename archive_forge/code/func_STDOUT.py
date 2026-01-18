import py
import sys
def STDOUT(msg):
    """ consumer that writes to sys.stdout """
    sys.stdout.write(str(msg) + '\n')