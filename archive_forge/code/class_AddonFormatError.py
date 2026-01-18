import base64
import copy
import json
import os
import re
import shutil
import sys
import tempfile
import warnings
import zipfile
from io import BytesIO
from xml.dom import minidom
from typing_extensions import deprecated
from selenium.common.exceptions import WebDriverException
@deprecated('Addons must be added after starting the session')
class AddonFormatError(Exception):
    """Exception for not well-formed add-on manifest files."""