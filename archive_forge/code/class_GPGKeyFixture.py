import email
import email.errors
import os
import re
import sysconfig
import tempfile
import textwrap
import fixtures
import pkg_resources
import six
import testscenarios
import testtools
from testtools import matchers
import virtualenv
from wheel import wheelfile
from pbr import git
from pbr import packaging
from pbr.tests import base
class GPGKeyFixture(fixtures.Fixture):
    """Creates a GPG key for testing.

    It's recommended that this be used in concert with a unique home
    directory.
    """

    def setUp(self):
        super(GPGKeyFixture, self).setUp()
        tempdir = self.useFixture(fixtures.TempDir())
        gnupg_version_re = re.compile('^gpg\\s.*\\s([\\d+])\\.([\\d+])\\.([\\d+])')
        gnupg_version = base._run_cmd(['gpg', '--version'], tempdir.path)
        for line in gnupg_version[0].split('\n'):
            gnupg_version = gnupg_version_re.match(line)
            if gnupg_version:
                gnupg_version = (int(gnupg_version.group(1)), int(gnupg_version.group(2)), int(gnupg_version.group(3)))
                break
        else:
            if gnupg_version is None:
                gnupg_version = (0, 0, 0)
        config_file = os.path.join(tempdir.path, 'key-config')
        with open(config_file, 'wt') as f:
            if gnupg_version[0] == 2 and gnupg_version[1] >= 1:
                f.write('\n                %no-protection\n                %transient-key\n                ')
            f.write('\n            %no-ask-passphrase\n            Key-Type: RSA\n            Name-Real: Example Key\n            Name-Comment: N/A\n            Name-Email: example@example.com\n            Expire-Date: 2d\n            %commit\n            ')
        if gnupg_version[0] == 1:
            gnupg_random = '--quick-random'
        elif gnupg_version[0] >= 2:
            gnupg_random = '--debug-quick-random'
        else:
            gnupg_random = ''
        base._run_cmd(['gpg', '--gen-key', '--batch', gnupg_random, config_file], tempdir.path)