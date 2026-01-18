from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import gzip
import io
import json
import os
import string
import subprocess
import sys
import tarfile
import tempfile
import threading
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v1 import docker_creds as v1_creds
from containerregistry.client.v1 import docker_http
import httplib2
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
def _next_layer(self, sample, layer_byte_size, blob):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        if blob:
            info = tarfile.TarInfo(name='./' + self._next_id(sample))
            info.size = len(blob)
            tar.addfile(info, fileobj=io.BytesIO(blob))
        elif sys.platform.startswith('linux') and layer_byte_size >= 1024 * 1024:
            mb = layer_byte_size / (1024 * 1024)
            tempdir = tempfile.mkdtemp()
            data_filename = os.path.join(tempdir, 'a.bin')
            if os.path.exists(data_filename):
                os.remove(data_filename)
            process = subprocess.Popen(['dd', 'if=/dev/urandom', 'of=%s' % data_filename, 'bs=1M', 'count=%d' % mb])
            process.wait()
            with io.open(data_filename, u'rb') as fd:
                info = tar.gettarinfo(name=data_filename)
                tar.addfile(info, fileobj=fd)
                os.remove(data_filename)
                os.rmdir(tempdir)
        else:
            data = sample(string.printable.encode('utf8'), layer_byte_size)
            info = tarfile.TarInfo(name='./' + self._next_id(sample))
            info.size = len(data)
            tar.addfile(info, fileobj=io.BytesIO(data))
    return buf.getvalue()