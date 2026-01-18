import io
import sys
import time
import marshal
class fake_frame:

    def __init__(self, code, prior):
        self.f_code = code
        self.f_back = prior