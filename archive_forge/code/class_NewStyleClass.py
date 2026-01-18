import sys
from zope.interface.advice import getFrameInfo
class NewStyleClass:
    __metaclass__ = type
    classLevelFrameInfo = getFrameInfo(sys._getframe())