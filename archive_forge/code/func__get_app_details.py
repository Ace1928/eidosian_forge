import json
import logging
import os
import uuid
from http import HTTPStatus
from typing import Any, Dict, Iterator, List, Optional
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.pebblo import (
def _get_app_details(self) -> App:
    """Fetch app details. Internal method.

        Returns:
            App: App details.
        """
    framework, runtime = get_runtime()
    app = App(name=self.app_name, owner=self.owner, description=self.description, load_id=self.load_id, runtime=runtime, framework=framework, plugin_version=PLUGIN_VERSION)
    return app