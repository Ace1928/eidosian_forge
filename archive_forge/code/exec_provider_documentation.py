import json
import os
import subprocess
import sys
from .config_exception import ConfigException

        exec_config must be of type ConfigNode because we depend on
        safe_get(self, key) to correctly handle optional exec provider
        config parameters.
        