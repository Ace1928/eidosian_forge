from ast import (
import ast
import copy
from typing import Dict, Optional, Union

    Mangle given names in and ast tree to make sure they do not conflict with
    user code.
    