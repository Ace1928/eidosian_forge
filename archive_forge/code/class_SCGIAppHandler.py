import sys
import time
from scgi import scgi_server
class SCGIAppHandler(SWAP):

    def __init__(self, *args, **kwargs):
        self.prefix = prefix
        self.app_obj = application
        SWAP.__init__(self, *args, **kwargs)