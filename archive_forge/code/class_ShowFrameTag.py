import os
import zlib
import time  # noqa
import logging
import numpy as np
class ShowFrameTag(ControlTag):

    def __init__(self):
        ControlTag.__init__(self)
        self.tagtype = 1

    def process_tag(self):
        self.bytes = bytes()