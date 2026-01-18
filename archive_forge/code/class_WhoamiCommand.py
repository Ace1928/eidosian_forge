import subprocess
from argparse import ArgumentParser
from typing import List, Union
from huggingface_hub.hf_api import HfFolder, create_repo, whoami
from requests.exceptions import HTTPError
from . import BaseTransformersCLICommand
class WhoamiCommand(BaseUserCommand):

    def run(self):
        print(ANSI.red('WARNING! `transformers-cli whoami` is deprecated and will be removed in v5. Please use `huggingface-cli whoami` instead.'))
        token = HfFolder.get_token()
        if token is None:
            print('Not logged in')
            exit()
        try:
            user, orgs = whoami(token)
            print(user)
            if orgs:
                print(ANSI.bold('orgs: '), ','.join(orgs))
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)