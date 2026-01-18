import subprocess
from argparse import ArgumentParser
from typing import List, Union
from huggingface_hub.hf_api import HfFolder, create_repo, whoami
from requests.exceptions import HTTPError
from . import BaseTransformersCLICommand
class LoginCommand(BaseUserCommand):

    def run(self):
        print(ANSI.red('ERROR! `huggingface-cli login` uses an outdated login mechanism that is not compatible with the Hugging Face Hub backend anymore. Please use `huggingface-cli login instead.'))