import dataclasses
import json
from urllib.parse import urlencode, urlunparse
from jupyter_server.base.handlers import APIHandler
from tornado import web
from jupyterlab.extensions.manager import ExtensionManager
POST query performs an action on a specific extension

        Body arguments:
            {
                "cmd": Action to perform - ["install", "uninstall", "enable", "disable"]
                "extension_name": Extension name
                "extension_version": [optional] Extension version (used only for install action)
            }
        