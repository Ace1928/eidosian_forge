import time
from twisted.internet import defer
from twisted.spread import banana, flavors, jelly
def _pubReady(result, d2):
    """(internal)"""
    result.callWhenActivated(d2.callback)