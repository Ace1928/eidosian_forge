import os
import zlib
import time  # noqa
import logging
import numpy as np
class FileAttributesTag(ControlTag):

    def __init__(self):
        ControlTag.__init__(self)
        self.tagtype = 69

    def process_tag(self):
        self.bytes = '\x00'.encode('ascii') * (1 + 3)