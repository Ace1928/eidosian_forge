import argparse
import io
import json
import os
from unittest import mock
import subprocess
import tempfile
import testtools
from glanceclient import exc
from glanceclient import shell
import glanceclient.v1.client as client
import glanceclient.v1.images
import glanceclient.v1.shell as v1shell
from glanceclient.tests import utils
def _fake_update_func(self, *args, **kwargs):
    """Replace glanceclient.images.update with a fake.

        To determine the parameters that would be supplied with the update
        request.
        """
    self.collected_args = (args, kwargs)
    return args[0]