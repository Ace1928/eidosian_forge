import collections
import platform
import sys
def _platform_tuple():
    try:
        p_system = platform.system()
        p_release = platform.release()
    except IOError:
        p_system = 'Unknown'
        p_release = 'Unknown'
    return (p_system, p_release)