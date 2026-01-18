import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union, cast
from langsmith import env as ls_env
from langsmith import utils as ls_utils
def _start_local(self) -> None:
    command = [*self.docker_compose_command, '-f', str(self.docker_compose_file)]
    subprocess.run([*command, 'up', '--quiet-pull', '--wait'])
    logger.info('LangSmith server is running at http://localhost:80/api.\nTo view the app, navigate your browser to http://localhost:80\n\nTo connect your LangChain application to the server locally,\nset the following environment variable when running your LangChain application.\n')
    logger.info('\tLANGSMITH_TRACING=true')
    logger.info('\tLANGSMITH_ENDPOINT=http://localhost:80/api\n')
    self._open_browser('http://localhost')