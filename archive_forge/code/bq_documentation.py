from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pdb
import sys
import traceback
from absl import flags
from pyglib import appcommands
import bigquery_client
import bq_flags
import bq_utils
import credential_loader
from clients import utils as bq_client_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from frontend import commands
from frontend import commands_iam
from frontend import command_copy
from frontend import command_delete
from frontend import command_extract
from frontend import command_info
from frontend import command_list
from frontend import command_load
from frontend import command_make
from frontend import command_mkdef
from frontend import command_query
from frontend import command_repl
from frontend import command_show
from frontend import command_truncate
from frontend import command_update
from frontend import utils as frontend_utils
from utils import bq_id_utils
Function to be used as setuptools script entry point.

  Appcommands assumes that it always runs as __main__, but launching
  via a setuptools-generated entry_point breaks this rule. We do some
  trickery here to make sure that appcommands and flags find their
  state where they expect to by faking ourselves as __main__.
  