from __future__ import annotations
import abc
import dataclasses
import json
import os
import re
import stat
import traceback
import uuid
import time
import typing as t
from .http import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .ci import (
from .data import (
class SshKey:
    """Container for SSH key used to connect to remote instances."""
    KEY_TYPE = 'rsa'
    KEY_NAME = f'id_{KEY_TYPE}'
    PUB_NAME = f'{KEY_NAME}.pub'

    @mutex
    def __init__(self, args: EnvironmentConfig) -> None:
        key_pair = self.get_key_pair()
        if not key_pair:
            key_pair = self.generate_key_pair(args)
        key, pub = key_pair
        key_dst, pub_dst = self.get_in_tree_key_pair_paths()

        def ssh_key_callback(payload_config: PayloadConfig) -> None:
            """
            Add the SSH keys to the payload file list.
            They are either outside the source tree or in the cache dir which is ignored by default.
            """
            files = payload_config.files
            permissions = payload_config.permissions
            files.append((key, os.path.relpath(key_dst, data_context().content.root)))
            files.append((pub, os.path.relpath(pub_dst, data_context().content.root)))
            permissions[os.path.relpath(key_dst, data_context().content.root)] = stat.S_IRUSR | stat.S_IWUSR
        data_context().register_payload_callback(ssh_key_callback)
        self.key, self.pub = (key, pub)
        if args.explain:
            self.pub_contents = None
            self.key_contents = None
        else:
            self.pub_contents = read_text_file(self.pub).strip()
            self.key_contents = read_text_file(self.key).strip()

    @staticmethod
    def get_relative_in_tree_private_key_path() -> str:
        """Return the ansible-test SSH private key path relative to the content tree."""
        temp_dir = ResultType.TMP.relative_path
        key = os.path.join(temp_dir, SshKey.KEY_NAME)
        return key

    def get_in_tree_key_pair_paths(self) -> t.Optional[tuple[str, str]]:
        """Return the ansible-test SSH key pair paths from the content tree."""
        temp_dir = ResultType.TMP.path
        key = os.path.join(temp_dir, self.KEY_NAME)
        pub = os.path.join(temp_dir, self.PUB_NAME)
        return (key, pub)

    def get_source_key_pair_paths(self) -> t.Optional[tuple[str, str]]:
        """Return the ansible-test SSH key pair paths for the current user."""
        base_dir = os.path.expanduser('~/.ansible/test/')
        key = os.path.join(base_dir, self.KEY_NAME)
        pub = os.path.join(base_dir, self.PUB_NAME)
        return (key, pub)

    def get_key_pair(self) -> t.Optional[tuple[str, str]]:
        """Return the ansible-test SSH key pair paths if present, otherwise return None."""
        key, pub = self.get_in_tree_key_pair_paths()
        if os.path.isfile(key) and os.path.isfile(pub):
            return (key, pub)
        key, pub = self.get_source_key_pair_paths()
        if os.path.isfile(key) and os.path.isfile(pub):
            return (key, pub)
        return None

    def generate_key_pair(self, args: EnvironmentConfig) -> tuple[str, str]:
        """Generate an SSH key pair for use by all ansible-test invocations for the current user."""
        key, pub = self.get_source_key_pair_paths()
        if not args.explain:
            make_dirs(os.path.dirname(key))
        if not os.path.isfile(key) or not os.path.isfile(pub):
            run_command(args, ['ssh-keygen', '-m', 'PEM', '-q', '-t', self.KEY_TYPE, '-N', '', '-f', key], capture=True)
            if args.explain:
                return (key, pub)
            key_contents = read_text_file(key)
            key_contents = re.sub('(BEGIN|END) PRIVATE KEY', '\\1 RSA PRIVATE KEY', key_contents)
            write_text_file(key, key_contents)
        return (key, pub)