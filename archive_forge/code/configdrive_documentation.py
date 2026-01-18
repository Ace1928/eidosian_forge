import base64
import contextlib
import gzip
import json
import os
import shutil
import subprocess
import tempfile
import typing as ty
Pack a directory with files into a Bare Metal service configdrive.

    Creates an ISO image with the files and label "config-2".

    :param str path: Path to directory with files
    :return: configdrive contents as a base64-encoded string.
    