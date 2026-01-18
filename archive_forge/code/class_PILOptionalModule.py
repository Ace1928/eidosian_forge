import importlib
import logging
class PILOptionalModule(OptionalModule):

    def __init__(self, failMessage=None, require=False):
        super(PILOptionalModule, self).__init__('PIL', failMessage=failMessage, require=require)