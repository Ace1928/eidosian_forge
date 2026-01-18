import json
import os
import subprocess
import sys
import warnings
from argparse import ArgumentParser
from contextlib import AbstractContextManager
from typing import Dict, List, Optional
import requests
from ..utils import logging
from . import BaseTransformersCLICommand
class LfsCommands(BaseTransformersCLICommand):
    """
    Implementation of a custom transfer agent for the transfer type "multipart" for git-lfs. This lets users upload
    large files >5GB 🔥. Spec for LFS custom transfer agent is:
    https://github.com/git-lfs/git-lfs/blob/master/docs/custom-transfers.md

    This introduces two commands to the CLI:

    1. $ transformers-cli lfs-enable-largefiles

    This should be executed once for each model repo that contains a model file >5GB. It's documented in the error
    message you get if you just try to git push a 5GB file without having enabled it before.

    2. $ transformers-cli lfs-multipart-upload

    This command is called by lfs directly and is not meant to be called by the user.
    """

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        enable_parser = parser.add_parser('lfs-enable-largefiles', help='Deprecated: use `huggingface-cli` instead. Configure your repository to enable upload of files > 5GB.')
        enable_parser.add_argument('path', type=str, help='Local path to repository you want to configure.')
        enable_parser.set_defaults(func=lambda args: LfsEnableCommand(args))
        upload_parser = parser.add_parser(LFS_MULTIPART_UPLOAD_COMMAND, help='Deprecated: use `huggingface-cli` instead. Command will get called by git-lfs, do not call it directly.')
        upload_parser.set_defaults(func=lambda args: LfsUploadCommand(args))