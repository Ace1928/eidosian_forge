import logging
import os
import re
class UNSIGNED:

    def __copy__(self):
        return self

    def __deepcopy__(self, memodict):
        return self