import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
def _make_nccl_url(public_version, filename):
    return 'https://developer.download.nvidia.com/compute/redist/nccl/' + 'v{}/{}'.format(public_version, filename)