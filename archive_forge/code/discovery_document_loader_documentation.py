import pkgutil
from typing import Optional
from absl import logging
Loads a discovery doc as `bytes` specified by `package` and `resource` returning None on error.