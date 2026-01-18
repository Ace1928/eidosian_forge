import os
import platform
import sys
from typing import List
class XGBoostLibraryNotFound(Exception):
    """Error thrown by when xgboost is not found"""