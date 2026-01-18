from collections import OrderedDict
from twisted.trial import unittest
from twisted.words.xish import utility
from twisted.words.xish.domish import Element
from twisted.words.xish.utility import EventDispatcher
class OrderedCallbackList(utility.CallbackList):

    def __init__(self):
        self.callbacks = OrderedDict()