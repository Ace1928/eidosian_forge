from __future__ import (absolute_import, division, print_function)
import collections
import datetime
import functools
import hashlib
import json
import os
import stat
import tarfile
import time
import threading
from http import HTTPStatus
from http.client import BadStatusLine, IncompleteRead
from urllib.error import HTTPError, URLError
from urllib.parse import quote as urlquote, urlencode, urlparse, parse_qs, urljoin
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.galaxy.user_agent import user_agent
from ansible.module_utils.api import retry_with_delays_and_condition
from ansible.module_utils.api import generate_jittered_backoff
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.urls import open_url, prepare_multipart
from ansible.utils.display import Display
from ansible.utils.hashing import secure_hash_s
from ansible.utils.path import makedirs_safe
class CollectionVersionMetadata:

    def __init__(self, namespace, name, version, download_url, artifact_sha256, dependencies, signatures_url, signatures):
        """
        Contains common information about a collection on a Galaxy server to smooth through API differences for
        Collection and define a standard meta info for a collection.

        :param namespace: The namespace name.
        :param name: The collection name.
        :param version: The version that the metadata refers to.
        :param download_url: The URL to download the collection.
        :param artifact_sha256: The SHA256 of the collection artifact for later verification.
        :param dependencies: A dict of dependencies of the collection.
        :param signatures_url: The URL to the specific version of the collection.
        :param signatures: The list of signatures found at the signatures_url.
        """
        self.namespace = namespace
        self.name = name
        self.version = version
        self.download_url = download_url
        self.artifact_sha256 = artifact_sha256
        self.dependencies = dependencies
        self.signatures_url = signatures_url
        self.signatures = signatures