import subprocess
from argparse import ArgumentParser
from typing import List, Union
from huggingface_hub.hf_api import HfFolder, create_repo, whoami
from requests.exceptions import HTTPError
from . import BaseTransformersCLICommand
class LogoutCommand(BaseUserCommand):

    def run(self):
        print(ANSI.red('ERROR! `transformers-cli logout` uses an outdated logout mechanism that is not compatible with the Hugging Face Hub backend anymore. Please use `huggingface-cli logout instead.'))