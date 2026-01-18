from __future__ import annotations
import os
import pkgutil
import warnings
import zipfile
from abc import ABCMeta
from base64 import b64decode
from base64 import encodebytes
from hashlib import md5 as md5_hash
from io import BytesIO
from typing import List
from selenium.common.exceptions import JavascriptException
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.utils import keys_to_typing
from .command import Command
from .shadowroot import ShadowRoot
def _load_js():
    global getAttribute_js
    global isDisplayed_js
    _pkg = '.'.join(__name__.split('.')[:-1])
    getAttribute_js = pkgutil.get_data(_pkg, 'getAttribute.js').decode('utf8')
    isDisplayed_js = pkgutil.get_data(_pkg, 'isDisplayed.js').decode('utf8')