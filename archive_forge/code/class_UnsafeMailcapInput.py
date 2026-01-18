import os
import warnings
import re
class UnsafeMailcapInput(Warning):
    """Warning raised when refusing unsafe input"""