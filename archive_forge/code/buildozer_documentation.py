import sys
import logging
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile
import PySide6
from pathlib import Path
from typing import List
from pkginfo import Wheel
from .. import MAJOR_VERSION, BaseConfig, Config, run_command
from . import (create_recipe, find_lib_dependencies, find_qtlibs_in_wheel,

        Given pysidedeploy_config.modules, find all the other dependent Qt modules. This is
        done by using llvm-readobj (readelf) to find the dependent libraries from the module
        library.
        